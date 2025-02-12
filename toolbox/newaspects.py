import os
import json
import re
import requests
from PIL import Image
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from io import BytesIO
import matplotlib.pyplot as plt
import streamlit as st

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


# -----------------------------
# Send a request to OpenAI
def send_openai_prompt(prompt_text, model_name="gpt-4o", temperature=0.7):
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text[:10000]}],
            model=model_name,
            temperature=temperature,
            # max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Request failed: {e}"


# -----------------------------
# Extract an image URL from a webpage
def image_url_finder(webpage_url):
    try:
        response = requests.get(webpage_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for img in soup.find_all('meta'):
            img_url = img.get('content')
            if img_url and re.search(r'(\.jpg|\.jpeg|\.png|\.webp)$', img_url, re.IGNORECASE):
                return img_url
        return None
    except Exception as e:
        print(f"Failed to retrieve image from {webpage_url}: {e}")
        return None


# -----------------------------
# Extract URLs from knowledge graph text
def find_urls(kg_text):
    return re.findall(r'https:\/\/www\.inaturalist\.org\/observations\/\d+', str(kg_text))


# -----------------------------
# UPDATED: Extract structured elements from the user's question.
def extract_keywords_from_question(question: str) -> dict:
    """
    Extract from the user's question the following elements:
      - "species": one of [Fish, Reptile, Amphibian, Birds]
      - "county": the county name mentioned (if any)
      - "target": the wildlife management task, one of ["Data Display", "Data Analysis", "Conservation Management"]
    
    The LLM should output a valid JSON-formatted dictionary. If extraction fails, defaults are returned.
    """
    prompt = f"""
You are an intelligent assistant. From the user's question below, extract the following elements and output a valid JSON-formatted dictionary:
- "species": one of the following (Fish, Reptile, Amphibian, Birds)
- "county": the county name mentioned (if any)
- "target": the wildlife management task, one of: "Data Display", "Data Analysis", "Conservation Management"

User question:
{question}

Output a JSON dictionary with keys "species", "county", and "target". For example:
{{ "species": "Birds", "county": "Orange", "target": "Data Display" }}

If you cannot determine these elements, please return:
{{ "species": "Reptile", "county": "Unknown", "target": "Data Display" }}
    """
    response_text = send_openai_prompt(prompt)
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        result = {"species": "Reptile", "county": "Unknown", "target": "Data Display"}
    return result


# -----------------------------
# UPDATED: Construct a Cypher query using the structured query elements.
def construct_cypher_query(structured_dict, multi_option) -> str:
    """
    Build a Cypher query based on the structured dictionary.
    structured_dict should include:
      - "species": e.g., "Fish", "Reptile", "Amphibian", or "Birds"
      - "county": the county name (e.g., "Orange")
      - "target": the task (e.g., "Data Display")
      
    The species value is mapped to a node label (e.g., "Birds" -> "Bird_name").
    If multi_option equals 1, a query that returns selected fields is built.
    """
    species = structured_dict.get("species", "Reptile")
    county = structured_dict.get("county", "Unknown")
    target = structured_dict.get("target", "Data Display")
    
    # Map species to a node label
    label_map = {
       "Fish": "Fish_name",
       "Reptile": "Reptile_name",
       "Amphibian": "Amphibian_name",
       "Birds": "Bird_name"
    }
    species_label = label_map.get(species, "Reptile_name")
    
    # Build a basic Cypher query. Assume there is a County node with a 'name' property.
    if multi_option == 1:
        cypher_query = f"""
        MATCH (s:{species_label})-[:OBSERVED_AT]->(c:County)
        WHERE c.name = '{county}'
        RETURN s.name AS species_name, s.multimedia AS multimedia, c.name AS county_name
        """
    else:
        cypher_query = f"""
        MATCH (s:{species_label})-[:OBSERVED_AT]->(c:County)
        WHERE c.name = '{county}'
        RETURN s, c
        """
    return cypher_query


# -----------------------------
# Query the Neo4j graph database
def query_neo4j(cypher_query):
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            return [record for record in result]
    except Exception as e:
        print(f"Neo4j Query Failed: {e}")
        return []


# -----------------------------
# UPDATED: Process the question, query the graph, and generate the final answer.
def generate_aspect_chain(question, multimedia_option, llm_name, aspect=None, vllm_name=None):
    # Step 1: Extract structured elements from the user's question.
    structured_elements = extract_keywords_from_question(question)
    print("Extracted elements:", structured_elements)
    
    # Step 2: Construct a Cypher query using these elements.
    formatted_query = construct_cypher_query(structured_elements, multi_option=0)
    print("Formatted Cypher query:", formatted_query)
    
    # Step 3: Execute the Cypher query to get knowledge graph data.
    kg_output = query_neo4j(formatted_query)
    # Optionally limit output size
    kg_output = kg_output[:10000]
    
    # Step 4: Use the knowledge graph output to generate the final answer.
    prompt = f"""
    Use the following knowledge graph output to answer the user's question.
    
    Question: {question}
    Aspect: {aspect}
    KG Output: {kg_output}
    """
    response = send_openai_prompt(prompt)
    print("LLM Response:", response)
    print("KG Output:", kg_output)
    
    # Step 5 (optional): If the output includes media URLs, further process them.
    # (This section is commented out; you can enable if needed.)
    # if multimedia_option:
    #     if "http" in str(kg_output):
    #         urls = find_urls(kg_output)
    #         images = []
    #         for url in urls[:10]:
    #             image_url = image_url_finder(url)
    #             if image_url:
    #                 response = requests.get(image_url)
    #                 img = Image.open(BytesIO(response.content)).convert("RGB")
    #                 plt.imshow(img)
    #                 plt.show()
    #                 images.append(img)
    #         return {"answer": response, "URLs": urls}
    
    return {"answer": response}


# -----------------------------
# Generate a ticket (unchanged)
def generate_ticket(title, body):
    prompt = f"""
    Generate a new question for the knowledge graph based on the following example:

    Title: {title}
    Body: {body[:200]}

    Create a similar question in the same style.
    """
    result = send_openai_prompt(prompt)
    return result
