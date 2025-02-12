import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests
import re
from toolbox.aspects import send_openai_prompt
import streamlit as st

# Neo4j configuration
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -------------------------------------------------------------------
# New helper function: extract the species from the question.
def extract_species_from_question(question: str) -> str:
    """
    From the question, determine the species among the following:
    Fish, Reptile, Amphibian, Birds.
    Return only the species name as plain text.
    """
    prompt = f"""
From the following question, determine the species among: Fish, Reptile, Amphibian, Birds.
Question: {question}
Return only the species name as plain text.
    """
    response = send_openai_prompt(prompt)
    species = response.strip()
    if species not in ["Fish", "Reptile", "Amphibian", "Birds"]:
        species = "Reptile"  # default if not clearly determined
    return species

# -------------------------------------------------------------------
# 1. Infer node scope from question using the new structured approach.
def infer_node_scope_from_question(llm, question):
    """
    Instead of using a free-form prompt, we now extract the species from the question
    and map it to the corresponding Neo4j node label.
    """
    species = extract_species_from_question(question)
    label_map = {
       "Fish": "Fish_name",
       "Reptile": "Reptile_name",
       "Amphibian": "Amphibian_name",
       "Birds": "Bird_name"
    }
    return label_map.get(species, "Reptile_name")

# -------------------------------------------------------------------
# 2. Construct a Cypher query with the given node scope.
def construct_cypher_query_with_scope(node_scope):
    """
    Build a Cypher query to fetch Wikidata IDs from nodes with the specified label.
    """
    cypher_query = f"""
    MATCH (n:{node_scope})
    WHERE n.wikidata_id IS NOT NULL 
    RETURN n.wikidata_id AS wikidata_id
    LIMIT 5
    """
    return cypher_query

# -------------------------------------------------------------------
# 3. Execute Neo4j query to fetch Wikidata IDs.
def fetch_wikidata_ids_from_neo4j(node_scope):
    query = f"""
    MATCH (n:{node_scope})
    WHERE n.wikidata_id IS NOT NULL 
    RETURN n.wikidata_id AS wikidata_id
    LIMIT 2
    """
    with driver.session() as session:
        result = session.run(query)
        return [record['wikidata_id'] for record in result]

# -------------------------------------------------------------------
# 4. Retrieve webpage content from a given URL.
def retrieve_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Will raise an exception if status is not 200
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve content from {url}: {e}")
        return None

# -------------------------------------------------------------------
# 5. Judgement function: check if the question appears in the content.
def judgement(question, content):
    for text in content:
        if question.lower() in text.lower():
            return f"Based on the content, the answer to '{question}' is found."
    return "No"

# -------------------------------------------------------------------
# 6. Extract top K links from a webpage.
def extract_top_k_links(weburl, topK):
    try:
        response = requests.get(weburl, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        # Filter for links starting with http
        relevant_links = [link['href'] for link in links if link['href'].startswith('http')][:topK]
        return relevant_links
    except requests.exceptions.RequestException as e:
        print(f"Failed to extract links from {weburl}: {e}")
        return []

# -------------------------------------------------------------------
# 7. Web search agent main function.
def web_search_agent(llm, question, node_scope=""):
    """
    This function first fetches Wikidata IDs from Neo4j based on the inferred node scope.
    Then, it constructs Wikidata links and retrieves the page content.
    """
    # Step 1: Infer node scope if not provided.
    if not node_scope:
        node_scope = infer_node_scope_from_question(llm, question)
    
    # Step 2: Fetch Wikidata IDs from Neo4j.
    wikidata_ids = fetch_wikidata_ids_from_neo4j(node_scope)
    if not wikidata_ids:
        return "No relevant Wikidata IDs found in the knowledge graph."
    
    # Step 3: Build Wikidata links.
    wikidata_links = [f"https://www.wikidata.org/wiki/{wid}" for wid in wikidata_ids]
    
    # Step 4: Retrieve content from each Wikidata link.
    all_content = []
    for link in wikidata_links:
        print(f"Accessing: {link}")
        content = retrieve_content(link)
        if content:
            all_content.append(content)
    
    print("Collected content:", all_content)
    return all_content

# -------------------------------------------------------------------
# Example usage (if needed, you can call web_search_agent from your Streamlit app):
if __name__ == "__main__":
    # For testing outside of Streamlit:
    test_question = "What are some observations of birds in Florida?"
    # Here we assume 'llm' is available; you may pass None or a dummy variable since our functions use send_openai_prompt.
    node_scope = infer_node_scope_from_question(None, test_question)
    print("Inferred node scope:", node_scope)
    query = construct_cypher_query_with_scope(node_scope)
    print("Constructed Cypher query:", query)
    wikidata_ids = fetch_wikidata_ids_from_neo4j(node_scope)
    print("Fetched Wikidata IDs:", wikidata_ids)
    contents = web_search_agent(None, test_question, node_scope)
    print("Retrieved content from Wikidata pages:", contents)
