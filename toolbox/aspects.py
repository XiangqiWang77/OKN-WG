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

# 加载环境变量
#load_dotenv()
import streamlit as st
# OpenAI 配置
api_key = st.secrets["OPENAI_KEY"]
client = OpenAI(api_key=api_key)

# Neo4j 数据库配置
# Neo4j 数据库配置
NEO4J_URI=st.secrets["NEO4J_URI"]
NEO4J_USER="neo4j"
NEO4J_PASSWORD=st.secrets["NEO4J_PASSWORD"]
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# 发送请求到 OpenAI
def send_openai_prompt(prompt_text, model_name="gpt-4o", temperature=0.7):
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text[:10000]}],
            model=model_name,
            temperature=temperature,
            #max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Request failed: {e}"


# 提取网页上的图片 URL
def image_url_finder(webpage_url):
    response = requests.get(webpage_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for img in soup.find_all('meta'):
        img_url = img.get('content')
        if img_url and re.search(r'(\.jpg|\.jpeg|\.png|\.webp)$', img_url, re.IGNORECASE):
            return img_url
    return None


# 从知识图谱查询结果中提取 URL
def find_urls(kg_text):
    return re.findall(r'https:\/\/www\.inaturalist\.org\/observations\/\d+', str(kg_text))


# 解析用户问题，提取关键词




def extract_node_categories_from_question(question: str) -> dict:
    """
    Extract the source and target node categories (e.g., "Bird_name", "Location")
    from the user's question using an LLM prompt.

    If extraction fails, it returns default node categories.
    """
    prompt = f"""
    You are an intelligent assistant responsible for deducing and extracting the "source node category"
    and "target node category" that we need to use in a Neo4j database based on the user's question.

    User question:
    {question}

    You must output a dictionary as a valid JSON-formatted string that can be loaded by 'json.loads':
    {{
        "source_node_category": "xxx",
        "target_node_category": "xxx"
    }}

    If you cannot determine these categories, please return:
    {{
        "source_node_category": "Reptile_name",
        "target_node_category": "Location"
    }}
    """

    # Here, replace 'send_openai_prompt' with your actual function for calling the LLM.
    # The function should return the raw string produced by the model (e.g., JSON).
    response_text = send_openai_prompt(prompt)

    try:
        categories = json.loads(response_text)
    except json.JSONDecodeError:
        # If parsing fails, return default categories
        categories = {
            "source_node_category": "Reptile_name",
            "target_node_category": "Location"
        }

    return categories


def construct_cypher_query(node_categories: dict, multi_option: int = 1) -> str:
    """
    Construct a Cypher query using the extracted node categories.

    node_categories: A dict in the form:
    {
        "source_node_category": "Bird_name",
        "target_node_category": "Location"
    }

    multi_option: Used to demonstrate returning different fields.
    """
    source_node_category = node_categories.get("source_node_category", "Reptile_name")
    target_node_category = node_categories.get("target_node_category", "Location")

    # Basic MATCH
    cypher_query = f"""
    MATCH (s:{source_node_category})-[r:OBSERVED_AT]->(t:{target_node_category})
    """

    # Return fields based on multi_option
    if multi_option == 1:
        cypher_query += """
    RETURN s.name AS source_name, r.multimedia AS multimedia, t.name AS target_name
        """
    else:
        cypher_query += """
    RETURN s, r, t
        """

    return cypher_query

# 查询 Neo4j 图数据库
def query_neo4j(cypher_query):
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            return [record for record in result]
    except Exception as e:
        print(f"Neo4j Query Failed: {e}")
        return []


# 处理查询和 LLM 交互，生成最终答案
def generate_aspect_chain(question, multimedia_option, llm_name, aspect=None, vllm_name=None):
    # 步骤1：从用户问题中提取关键词
    kwargs = extract_keywords_from_question(question)
    
    print(kwargs)
    # 步骤2：根据 aspect 调整查询方式
    #if aspect == "Multimedia":
    #    multi_option = 1  # 启用多媒体查询
    #else:
    #    multi_option = 0  # 关闭多媒体查询，返回一般结果
    
    # 步骤3：构建 Cypher 查询
    formatted_query = construct_cypher_query(kwargs, multi_option=0)
    
    print("formatted_query", formatted_query)


    # 步骤4：执行 Cypher 查询获取知识图谱数据
    kg_output = query_neo4j(formatted_query)

    kg_output=kg_output[:10000]
    
    # 步骤5：根据知识图谱输出，调用 LLM 生成答案
    prompt = f"""
    Use the following knowledge graph output to answer the user's question:
    - Question: {question}
    - Aspect: {aspect}
    - KG Output: {kg_output}
    """
    response = send_openai_prompt(prompt)

    print(response)

    print("kg_output", kg_output)

    # 步骤6：如果结果中包含多媒体 URL，进一步提取和展示
    #if multi_option:
    #    if "http" in str(kg_output):
    #        urls = find_urls(kg_output)
    #        images = []
    #        for url in urls[:10]:
    #            image_url = image_url_finder(url)
    #            if image_url:
    #                response = requests.get(image_url)
    #                img = Image.open(BytesIO(response.content)).convert("RGB")
    #                plt.imshow(img)
    #                plt.show()
    #                images.append(img)
    #        return {"answer": response, "URLs": urls}

    return {"answer": response}



# 生成问答任务
def generate_ticket(title, body):
    prompt = f"""
    Generate a new question for the knowledge graph based on the following example:

    Title: {title}
    Body: {body[:200]}

    Create a similar question in the same style.
    """
    result = send_openai_prompt(prompt)
    return result
