
from neo4j import GraphDatabase

# 设置Neo4j数据库连接
# Neo4j 数据库配置
import streamlit as st
NEO4J_URI=st.secrets["NEO4J_URI"]
NEO4J_USER="neo4j"
NEO4J_PASSWORD=st.secrets["NEO4J_PASSWORD"]
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
import openai
from openai import AzureOpenAI
from openai import OpenAI
import os
from dotenv import load_dotenv
import traceback
from PIL import Image
import torchvision.transforms as transforms
#load_dotenv()
import base64,io
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.wikidata.org/wiki/"

def fetch_wikidata_page(wikidata_id):
    url = BASE_URL + wikidata_id
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None
    

def get_top_nodes_for_question(question, N=5):
    query = """
    MATCH (n)
    WHERE n.description CONTAINS $question
    RETURN n.wikidata_id AS wikidata_id
    LIMIT $limit
    """
    with driver.session() as session:
        result = session.run(query, question=question, limit=N)
        return [record["wikidata_id"] for record in result]
    
import os
import requests
from openai import OpenAI, AzureOpenAI

def get_gpt4_text_response(web_content, question, model, need_azure=True, system_prompt='You are a helpful assistant.', temperature=0.0):
    # Initialize the client (Azure or OpenAI, depending on need_azure flag)
    if need_azure:
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        base_url = os.getenv('AZURE_OPENAI_ENDPOINT')
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-06-01",
            azure_endpoint=base_url
        )
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('OPENAI_API_BASE_URL')
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    # Prepare the system and user prompts
    system_input = {
        "role": "system",
        "content": system_prompt
    }
    user_input = {
        "role": "user",
        "content": f"Here is some web content: {web_content}. Now, answer this question based on the content: {question}"
    }

    # Generate the response using the model with text input
    response = client.chat.completions.create(
        model=model,
        messages=[system_input, user_input],
        temperature=temperature
    )

    response_text = response.choices[0].message.content
    return response_text


def extract_external_links(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    links = []
    for link in soup.find_all("a", href=True):
        if link["href"].startswith("http"):
            links.append(link["href"])
    return links


def web_agent(early_stop_num=10):
    for wikidata_id in top_nodes:
    html_content = fetch_wikidata_page(wikidata_id)
    if html_content:
        answer = query_gpt4(html_content, question)
        if answer.lower().startswith("no") or not answer:
            external_links = extract_external_links(html_content)
            for link in external_links:
                external_content = requests.get(link).text
                if external_content:
                    answer = query_gpt4(external_content, question)
                    if answer and not answer.lower().startswith("no"):
                        print(f"Answer found on external link {link}: {answer}")
                        break
        else:
            print(f"Answer from GPT-4 for {wikidata_id}: {answer}")