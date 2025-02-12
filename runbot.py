import os
import streamlit as st
import json
from io import BytesIO
from PIL import Image
import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from toolbox.chains import (
    configure_llm_only_chain,
    prompt_cypher,
    configure_qa_rag_chain,
    generate_llava_output,
    classfic,
    web_agent_multimedia
)
from toolbox.utils import ImageDownloader, convert_to_base64
from toolbox.aspects import generate_aspect_chain, extract_keywords_from_question
from toolbox.web_agent import web_search_agent, infer_node_scope_from_question

# -----------------------------
# Basic configuration (adjust as needed)
# load_dotenv(".env")
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

api_key = st.secrets["OPENAI_KEY"]
client = OpenAI(api_key=api_key)

with open('aspects.json', 'r') as f:
    json_data = json.load(f)

def send_openai_prompt(prompt_text, model_name="gpt-4o", temperature=0.7):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text}],
            model=model_name,
            temperature=temperature,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Request failed: {e}"
# -----------------------------

# -----------------------------
# [NEW] Function to convert a raw question into multiple-choice options.
def convert_question_to_multiple_choice(question):
    """
    Uses OpenAI to convert the input question into a multiple-choice question with 4 candidate options.
    Returns a list of strings, e.g.: ["Option 1", "Option 2", "Option 3", "Option 4"]
    """
    prompt = f"""
Convert the following question into a multiple-choice question by generating 4 candidate options for the user to choose from.
Return the result in a JSON array format.
Question: {question}
Example: ["Option 1", "Option 2", "Option 3", "Option 4"]
"""
    response = send_openai_prompt(prompt, model_name="gpt-4", temperature=0.7)
    try:
        options = json.loads(response)
        if not isinstance(options, list):
            raise ValueError("Result is not a list")
        return options
    except Exception as e:
        st.error(f"Error converting to multiple-choice format, using original question instead. Error: {e}")
        return [question]
# -----------------------------

def get_autocomplete_suggestions(question):
    prompt = f"""
You are given a graph containing information about animals and locations. Complete the user's partially entered question.
Input: {question}
"""
    return send_openai_prompt(prompt)

def load_llm(llm_name: str):
    return lambda prompt: send_openai_prompt(prompt, model_name=llm_name)

llm = load_llm("gpt-4o")

def query_neo4j(cypher_query, params=None):
    with driver.session() as session:
        result = session.run(cypher_query, params or {})
        return [record for record in result]

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
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                raise ValueError(f"URL does not point to an image: {url}")
            image = Image.open(BytesIO(response.content)).convert("RGB")
            cols[i].image(image, caption=f"Image {i+1}", use_column_width=True)
        except Exception as e:
            cols[i].write(f"Failed to load image {i+1}: {e}")

def chat_input():
    user_input_text = st.text_input("What do you want to know about wildlife in the US?", key="user_input_text", value="")
    if user_input_text:
        suggestion = get_autocomplete_suggestions(user_input_text)
        st.markdown(f"**Suggestion:** {suggestion}")
        if st.button("Accept Suggestion"):
            user_input_text = suggestion
        response = llm(user_input_text)
        with st.chat_message("assistant"):
            st.write(response)

def mode_select() -> list:
    options = [
        "Regular Response",
        "AI as Translator to KN-Wildlife",
        "AI as Toolbox for Aspect-Based Question",
        "AI as an Online(Wikidata) Searching Agent"
    ]
    multimedia_options = ["Text Only", "Text and Images"]
    selected_multimedia_mode = st.radio("Select output mode", multimedia_options, horizontal=True)
    mode_selected = st.radio("Select external sources", options, horizontal=True)
    return [mode_selected, selected_multimedia_mode]

def extract_keywords_from_response(response_text):
    prompt = f"""
Extract the top 5 most relevant keywords or entities from the following text.
Focus on nouns, names, or important terms relevant to the context.

Text: {response_text}

Return the keywords as a Python list of strings.
"""
    try:
        keywords_response = send_openai_prompt(prompt, model_name="gpt-4", temperature=0.5)
        keywords = eval(keywords_response.strip())
        return keywords
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def get_wikidata_id(entity_name):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": entity_name
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get('search'):
            return data['search'][0]['id']
    except Exception as e:
        print(f"Error fetching Wikidata ID for '{entity_name}': {e}")
    return None

def validate_image_url(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("image/"):
            return True
    except Exception as e:
        print(f"Error validating image URL '{url}': {e}")
    return False

def search_images_from_keywords(keywords):
    image_urls = []
    wikidata_sparql_endpoint = "https://query.wikidata.org/sparql"
    for keyword in keywords[:5]:
        entity_id = get_wikidata_id(keyword)
        if not entity_id:
            continue
        query = f"""
        SELECT ?image WHERE {{
          wd:{entity_id} wdt:P18 ?image.
        }}
        LIMIT 1
        """
        headers = {
            "User-Agent": "MyImageFetcher/1.0 (https://yourapp.example/; youremail@example.com)"
        }
        try:
            response = requests.get(
                wikidata_sparql_endpoint,
                params={"query": query, "format": "json"},
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            for result in data["results"]["bindings"]:
                if "image" in result:
                    image_url = result["image"]["value"]
                    if validate_image_url(image_url, headers):
                        image_urls.append(image_url)
                    else:
                        print(f"Invalid image URL (skipped): {image_url}")
        except Exception as e:
            print(f"Error fetching image for keyword '{keyword}': {e}")
    return image_urls

def handle_chat_mode(mode_info, user_input):
    # mode_info is a list: [selected_mode, multimedia_mode]
    print(mode_info[0])
    result = None
    if mode_info[0] == "Regular Response":
        if mode_info[1] == "Text Only":
            result = configure_llm_only_chain(user_input)
        elif mode_info[1] == "Text and Images":
            result = configure_llm_only_chain(user_input)
            keywords = extract_keywords_from_response(result)
            image_urls = search_images_from_keywords(keywords)
            if image_urls:
                display_images(image_urls)
            else:
                st.write("No images found for the extracted keywords.")
    elif mode_info[0] == "AI as Translator to KN-Wildlife":
        translate_function = prompt_cypher(llm)
        temp_result = translate_function(user_input)
        print(temp_result)
        if mode_info[1] == "Text Only":
            rag_chain = configure_qa_rag_chain(
                llm,
                query=temp_result,
                Graph_url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PASSWORD
            )
            result = rag_chain
        elif mode_info[1] == "Text and Images":
            kg_output = query_neo4j(temp_result)
            print("kg_output", kg_output)
            feed_result = generate_llava_output(user_input, kg_output)
            result = feed_result["answer"]
            keywords = extract_keywords_from_response(result)
            image_urls = search_images_from_keywords(keywords)
            if image_urls:
                display_images(image_urls)
            else:
                st.write("No images found for the extracted keywords.")
    elif mode_info[0] == "AI as Toolbox for Aspect-Based Question":
        temp_chain = generate_aspect_chain(
            question=user_input,
            multimedia_option=mode_info[1],
            llm_name=llm,
            vllm_name=None
        )
        feed_result = temp_chain
        if mode_info[1] == "Text Only":
            result = feed_result["answer"]
        elif mode_info[1] == "Text and Images":
            result = feed_result["answer"]
            keywords = extract_keywords_from_response(result)
            image_urls = search_images_from_keywords(keywords)
            if image_urls:
                display_images(image_urls)
            else:
                st.write("No images found for the extracted keywords.")
    elif mode_info[0] == "AI as an Online(Wikidata) Searching Agent":
        print("Function calling")
        node_scope = infer_node_scope_from_question(llm, user_input)
        print("node scope is", node_scope)
        agent_result = web_search_agent(
            llm=llm,
            question=user_input,
            node_scope=node_scope
        )
        final_str = (
            f"Question: {user_input}. The web retrieval agent returns: {agent_result}. "
            "Please provide an answer."
        )
        print(final_str)
        if mode_info[1] == "Text Only":
            result = send_openai_prompt(final_str)
        elif mode_info[1] == "Text and Images":
            result = send_openai_prompt(final_str)
            keywords = extract_keywords_from_response(result)
            image_urls = search_images_from_keywords(keywords)
            if image_urls:
                display_images(image_urls)
            else:
                st.write("No images found for the extracted keywords.")
    if result is None:
        result = "No result generated. Please refine your question or try a different mode."
    return result

import time
import threading

def keep_neo4j_alive(interval=300):
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
    while True:
        time.sleep(1)
        st.experimental_rerun()

def start_keep_alive_tasks():
    neo4j_thread = threading.Thread(target=keep_neo4j_alive, daemon=True)
    neo4j_thread.start()
    streamlit_thread = threading.Thread(target=keep_streamlit_alive, daemon=True)
    streamlit_thread.start()

if "keep_alive_started" not in st.session_state:
    start_keep_alive_tasks()
    st.session_state["keep_alive_started"] = True

query_params = st.query_params
query = query_params.get("query", None)
if query:
    if isinstance(query, list):
        query = " ".join(query)
st.write(f"**Query from URL:** {query}")

st.title("Wildlife Knowledge Assistant 🐾")
st.write("A bot to assist you with wildlife knowledge and Neo4j-powered queries.")

if "show_intro" not in st.session_state:
    st.session_state["show_intro"] = False

if st.sidebar.button("ℹ️ What is this bot?"):
    st.session_state["show_intro"] = not st.session_state["show_intro"]

if st.session_state["show_intro"]:
    st.sidebar.markdown("""
    ### Welcome to Wildlife Knowledge Assistant 🐾
    This bot is designed to help you:
    - Query and visualize wildlife-related data using **Neo4j**.
    - Ask complex questions and receive detailed answers powered by **LLMs**.
    - Explore multimedia (text and images) information related to your queries.
    - Discover more about wildlife in the United States and beyond.

    **Features**:
    - **Interactive Modes**: Choose from various response modes (text-only, multimedia, etc.).
    - **Custom AI Models**: Powered by GPT-based LLMs and integrated APIs for a rich experience.
    - **Neo4j Database**: Provides real-time data querying and visualization.

    **How to use**:
    1. Type your question or choose one of the selection modes below.
    2. Select a response mode and hit submit.
    3. Explore the detailed results with optional images.

    Enjoy exploring the wildlife knowledge base! 🌿
    """)

# -----------------------------
# User input section with three input modes:
# "Free Text", "Multiple-Choice Mode", and "Advanced Selection Mode".
st.markdown("### Choose your input mode")
input_mode = st.radio("Select input mode", options=["Free Text", "Multiple-Choice Mode", "Advanced Selection Mode"], key="input_mode")

mode_info = mode_select()

if query:
    st.write(f"**Query from URL:** {query}")
    result = handle_chat_mode(mode_info, query)
    st.write(result)
else:
    if input_mode == "Multiple-Choice Mode":
        # Multiple-Choice Mode: enter raw text, generate options, and select one.
        raw_question = st.text_input("Enter your question (raw text)", key="raw_question")
        if raw_question:
            if st.button("Generate Options", key="generate_options"):
                options = convert_question_to_multiple_choice(raw_question)
                st.session_state["multiple_choice_options"] = options
        if "multiple_choice_options" in st.session_state:
            final_question = st.radio("Select the question you want to ask", st.session_state["multiple_choice_options"], key="final_question")
            if st.button("Submit Question", key="submit_mc"):
                result = handle_chat_mode(mode_info, final_question)
                st.write(result)
    elif input_mode == "Advanced Selection Mode":
        # Advanced Selection Mode:
        # 1. Select species (Fish, Reptile, Amphibian, Birds)
        # 2. Enter county name (location)
        # 3. Choose a start and an end date for the time period.
        species_options = ["Select a species", "Fish", "Reptile", "Amphibian", "Birds"]
        selected_species = st.selectbox("Select species", species_options, key="species")
        
        county_name = st.text_input("Enter county name", key="county")
        
        # Use date_input for start and end dates.
        start_date = st.date_input("Start Date", key="start_date")
        end_date = st.date_input("End Date", key="end_date")
        
        if st.button("Submit Advanced Query", key="submit_advanced"):
            query_parts = []
            if selected_species != "Select a species":
                query_parts.append(f"Species: {selected_species}")
            if county_name:
                query_parts.append(f"County: {county_name}")
            # Ensure both dates are selected; format them as strings.
            if start_date and end_date:
                query_parts.append(f"Time: {start_date} to {end_date}")
            structured_query = " | ".join(query_parts)
            result = handle_chat_mode(mode_info, structured_query)
            st.write(result)
    else:
        # Free Text Mode: simply enter your question.
        user_input_text = st.text_input("What would you like to know?", key="user_input_key")
        if user_input_text:
            result = handle_chat_mode(mode_info, user_input_text)
            st.write(result)
# -----------------------------
