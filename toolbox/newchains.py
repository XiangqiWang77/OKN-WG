import re 
import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import json
import streamlit as st

# Load environment variables if needed
# load_dotenv()

# OpenAI API configuration
api_key = st.secrets["OPENAI_KEY"]
client = OpenAI(api_key=api_key)

# Neo4j database configuration
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# SentenceTransformer embedding model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


# Send OpenAI prompt (unchanged)
def send_openai_prompt(prompt_text, model_name="gpt-4o", temperature=0.7):
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text}],
            model=model_name,
            temperature=temperature,
            # max_tokens=token_limit
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Request failed: {e}"


# Image URL finder (unchanged)
def image_url_finder(webpage_url):
    try:
        response = requests.get(webpage_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for img in soup.find_all('meta'):
            img_url = img.get('content')
            if img_url and ('large.jpg' in img_url or 'large.jpeg' in img_url):
                return img_url
        return None
    except Exception as e:
        print(f"Failed to retrieve image from {webpage_url}: {e}")
        return None


# [UPDATED] Generate Cypher query using the new structured query schema.
def prompt_cypher(llm):
    def generate_cypher(user_input):
        prompt = f"""
Convert the following structured query into a Neo4j Cypher query.

Structured Query: {user_input}

Graph Schema:
- Nodes: 'Amphibian_name', 'Bird_name', 'Reptile_name', 'Fish_name', 'County', 'Task'
- Relationships: 'OBSERVED_AT' (from animals to counties), 'ASSIGNED_FOR' (from animals to tasks)
- Relationship Properties: 'multimedia', 'observed_times', 'dates'

Provide the final query in plain text format.
"""
        return send_openai_prompt(prompt, model_name="gpt-4o")
    return generate_cypher


# Execute a Cypher query (unchanged)
def query_neo4j(cypher_query):
    with driver.session() as session:
        result = session.run(cypher_query)
        return [record for record in result]


# LLM-only chain: directly generate an answer (unchanged)
def configure_llm_only_chain(question):
    prompt = f"""
You are a helpful assistant that answers questions about wildlife in Florida.
If you don't know the answer, just say that you don't know. Don't make up an answer.

Question: {question}
"""
    return send_openai_prompt(prompt)


# Configure RAG mode query for the knowledge graph (unchanged)
def configure_qa_rag_chain(llm, query, Graph_url=None, username=None, password=None):
    try:
        if not isinstance(query, str):
            raise ValueError("Cypher query is not a string.")
        if Graph_url and username and password:
            temp_driver = GraphDatabase.driver(Graph_url, auth=(username, password))
            with temp_driver.session() as session:
                result = session.run(query)
                return [record for record in result]
        else:
            results = query_neo4j(query)
        if not results:
            return "No relevant data found in the knowledge graph."
        return results
    except Exception as e:
        return f"Failed to retrieve information: {e}"


# Extract URLs from knowledge graph text (unchanged)
def find_urls(kg_text):
    return re.findall(r'https:\/\/www\.inaturalist\.org\/observations\/\d+', str(kg_text))


# Classify the question to determine the dictionary scope (unchanged)
def classfic(user_input, json_data, llm):
    prompt = f"""
You are an assistant that classifies questions about animal observations in Florida.
Classify the following question into the correct dictionary scope:

Question: {user_input}
Available Dictionary Scopes: {json.dumps(json_data)}

Return only the relevant scope name.
"""
    return send_openai_prompt(prompt)


# Generate LLaVA-style output from KG query results (unchanged)
def generate_llava_output(question, kg_output):
    prompt = f"""
Use the following knowledge graph output to answer the user's question:

Question: {question}
KG Output: {kg_output}

If there are image URLs, display them as part of the answer.
"""
    answer = send_openai_prompt(prompt)
    if "http" in str(kg_output):
        urls = find_urls(kg_output)
        images = []
        for url in urls[:10]:
            image_url = image_url_finder(url)
            if image_url:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                images.append(img)
        return {"answer": answer, "URLs": urls}
    return {"answer": answer}


# Handle Wikidata web agent multimedia (unchanged)
def web_agent_multimedia(agent_output):
    urls = find_urls(agent_output)
    valid_urls = []
    images = []
    for url in urls[:10]:
        image_url = image_url_finder(url)
        if image_url:
            try:
                response = requests.get(image_url, timeout=5)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                images.append(img)
                valid_urls.append(image_url)
            except (requests.RequestException, Image.UnidentifiedImageError) as e:
                print(f"Skipping invalid image: {image_url}. Error: {e}")
    if not valid_urls:
        return {"answer": agent_output, "URLs": [], "message": "No valid images found."}
    return {"answer": agent_output, "URLs": valid_urls}
