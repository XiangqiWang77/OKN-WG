import os
import json
import re
import requests
from PIL import Image
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from neo4j import GraphDatabase
from io import BytesIO
import streamlit as st
from openai import OpenAI

# Load environment variables if needed
# load_dotenv()

# OpenAI configuration
api_key = st.secrets["OPENAI_KEY"]
client = OpenAI(api_key=api_key)

# Neo4j database configuration
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Send OpenAI request
def send_openai_prompt(prompt_text, model_name="gpt-4o", temperature=0.7):
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text[:10000]}],
            model=model_name,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Request failed: {e}"

# Execute Neo4j query
def query_neo4j(cypher_query, params=None):
    with driver.session() as session:
        result = session.run(cypher_query, params or {})
        return [record for record in result]

# (Optional) Display images
def display_images(urls):
    if not urls:
        st.write("No images available.")
        return
    cols = st.columns(len(urls))
    for i, url in enumerate(urls):
        try:
            headers = {
                "User-Agent": "MyImageFetcher/1.0 (https://yourapp.example/; youremail@example.com)",
                "Referer": "https://www.wikimedia.org/"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            if not response.headers.get("Content-Type", "").startswith("image/"):
                raise ValueError(f"URL does not point to an image: {url}")
            image = Image.open(BytesIO(response.content)).convert("RGB")
            cols[i].image(image, caption=f"Image {i+1}", use_column_width=True)
        except Exception as e:
            cols[i].write(f"Failed to load image {i+1}: {e}")

# Use LLM to fill in the query template (fill in placeholders only)
def generate_query_from_template(animal, county):
    # Determine the node label based on the animal category
    animal_map = {
        "Reptile": "Reptile_name",
        "Amphibian": "Amphibian_name",
        "Fish": "Fish_name",
        "Bird": "Bird_name"
    }
    node_label = animal_map.get(animal, "Reptile_name")
    # Neo4j query template, note that the Location node is used to match the county name
    query_template = (
        "MATCH (a:{node_label})-[:OBSERVED_AT]->(l:Location) "
        "WHERE toLower(l.name) CONTAINS toLower('{county}') "
        "RETURN a, l"
    )
    # Construct prompt to let the LLM output a JSON object to fill in the placeholders
    prompt = f"""
You are given the following Neo4j Cypher query template with placeholders:
---
Template: "MATCH (a:{{node_label}})-[:OBSERVED_AT]->(l:Location) WHERE toLower(l.name) CONTAINS toLower('{{county}}') RETURN a, l"
---
Fill in the placeholders using the provided values.
Animal: "{animal}" (use the corresponding node label)
County: "{county}"
Output only a valid JSON object with keys "node_label" and "county". For example:
{{"node_label": "Reptile_name", "county": "Alachua"}}
    """
    json_str = send_openai_prompt(prompt, model_name="gpt-4o", temperature=0.3)
    try:
        fill_data = json.loads(json_str)
    except Exception as e:
        fill_data = {"node_label": node_label, "county": county}
    if "node_label" not in fill_data:
        fill_data["node_label"] = node_label
    if "county" not in fill_data:
        fill_data["county"] = county
    filled_query = query_template.format(node_label=fill_data["node_label"], county=fill_data["county"])
    return filled_query

# If there is a task requirement, use LLM to analyze the queried data
def analyze_data_with_task(kg_data, task, animal, county):
    prompt = f"""
Based on the following Neo4j query results: {kg_data},
please provide a detailed analysis for the wildlife management task "{task}".
The data is from animal category "{animal}" observed in a county containing "{county}".
Output only the analysis result.
    """
    return send_openai_prompt(prompt, model_name="gpt-4o", temperature=0.5)

# Keep Neo4j and Streamlit alive (retain original logic)
def keep_neo4j_alive(interval=300):
    import time
    def query():
        with driver.session() as session:
            try:
                session.run("MATCH (n) RETURN count(n) LIMIT 1")
                print("Keep-alive query sent to Neo4j.")
            except Exception as e:
                print(f"Neo4j keep-alive query failed: {e}")
    while True:
        query()
        time.sleep(interval)

def keep_streamlit_alive():
    import time
    while True:
        time.sleep(1)
        st.experimental_rerun()

def start_keep_alive_tasks():
    import threading
    neo4j_thread = threading.Thread(target=keep_neo4j_alive, daemon=True)
    neo4j_thread.start()
    streamlit_thread = threading.Thread(target=keep_streamlit_alive, daemon=True)
    streamlit_thread.start()

if "keep_alive_started" not in st.session_state:
    start_keep_alive_tasks()
    st.session_state["keep_alive_started"] = True

# Page display
st.title("Wildlife Knowledge Assistant 🐾")
st.write("Query animal data using selection mode, with options for subsequent task analysis.")

if "show_intro" not in st.session_state:
    st.session_state["show_intro"] = False
if st.sidebar.button("ℹ️ What is this bot?"):
    st.session_state["show_intro"] = not st.session_state["show_intro"]
if st.session_state["show_intro"]:
    st.sidebar.markdown("""
    ### Welcome to Wildlife Knowledge Assistant 🐾
    This tool allows you to query by selecting the following parameters:
    - Enter the county name
    - Select the animal category (Reptile, Amphibian, Fish, Bird)
    If needed, you can choose a specific task (e.g., Data Analysis or Conservation Management),
    and the system will further analyze the queried data.
    
    Please ensure that the 'name' property of the Location node in the database matches the entered county,
    and that the corresponding animal node (e.g., Reptile_name) contains the correct animal name.
    
    Enjoy using the tool!
    """)

st.markdown("### Please select query conditions")
county_name = st.text_input("Please enter the county name", key="county")
animal_options = ["Select an animal", "Reptile", "Amphibian", "Fish", "Bird"]
selected_animal = st.selectbox("Please select the animal category", animal_options, key="animal")
task_options = ["No further task", "Data Analysis", "Conservation Management"]
selected_task = st.selectbox("Please select a task (optional)", task_options, key="task")

if st.button("Submit query", key="submit_query"):
    if selected_animal == "Select an animal" or county_name.strip() == "":
        st.write("Please correctly select an animal category and enter the county name.")
    else:
        # Construct structured query string (for display only)
        structured_query = f"Animal: {selected_animal} | County: {county_name}"
        if selected_task != "No further task":
            structured_query += f" | Task: {selected_task}"
        st.write("Structured query:", structured_query)
        # Use LLM to fill in the template and generate a Neo4j query
        neo4j_query = generate_query_from_template(selected_animal, county_name)
        st.write("Generated Neo4j query:", neo4j_query)
        # Execute the query
        kg_data = query_neo4j(neo4j_query)
        st.write("Query results:", kg_data)
        # If there is a task requirement, use LLM to analyze the data
        if selected_task != "No further task":
            analysis_result = analyze_data_with_task(kg_data, selected_task, selected_animal, county_name)
            st.write("Analysis result:", analysis_result)
