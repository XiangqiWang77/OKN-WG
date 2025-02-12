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

# （如有需要，可启用下行加载环境变量）
# load_dotenv()

# OpenAI 配置
api_key = st.secrets["OPENAI_KEY"]
client = OpenAI(api_key=api_key)

# Neo4j 数据库配置
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# 发送 OpenAI 请求
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

# 执行 Neo4j 查询
def query_neo4j(cypher_query, params=None):
    with driver.session() as session:
        result = session.run(cypher_query, params or {})
        return [record for record in result]

# （可选）显示图片
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

# 利用 LLM 根据模板填空生成 Cypher 查询
def generate_query_from_template(reptile, county):
    # 预定义查询模板，其中 {reptile} 和 {county} 为占位符
    query_template = (
        "MATCH (s:Reptile_name) "
        "WHERE s.name CONTAINS '{reptile}' "
        "MATCH (s)-[:OBSERVED_AT]->(c:County) "
        "WHERE toLower(c.name) CONTAINS toLower('{county}') "
        "RETURN s, c"
    )
    # 构造提示，要求 LLM 返回 JSON 对象填充占位符
    prompt = f"""
You are given the following Neo4j Cypher query template with two placeholders:
---
Template: "{query_template}"
---
Fill in the placeholders using the following values.
Reptile: "{reptile}"
County: "{county}"
Output only a JSON object with keys "reptile" and "county". For example:
{{"reptile": "Turtle", "county": "Alachua"}}
    """
    json_str = send_openai_prompt(prompt, model_name="gpt-4o", temperature=0.3)
    try:
        fill_data = json.loads(json_str)
        # 若 LLM 未返回相应字段，则使用用户输入
        filled_query = query_template.format(
            reptile=fill_data.get("reptile", reptile),
            county=fill_data.get("county", county)
        )
        return filled_query
    except Exception as e:
        # 若解析失败，则直接用用户输入填充
        return query_template.format(reptile=reptile, county=county)

# 如果有任务要求，则调用 LLM 分析查询结果
def analyze_data_with_task(kg_data, task, reptile, county):
    prompt = f"""
Based on the following Neo4j query results: {kg_data},
please provide a detailed analysis for the wildlife management task "{task}".
The data is from reptile "{reptile}" observed in a county containing "{county}".
Output only the analysis result.
    """
    return send_openai_prompt(prompt, model_name="gpt-4o", temperature=0.5)

# 保活 Neo4j 与 Streamlit（保持原有逻辑）
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

# 页面展示
st.title("Wildlife Knowledge Assistant 🐾")
st.write("A bot to assist you with wildlife knowledge and Neo4j-powered queries using selection mode only.")

if "show_intro" not in st.session_state:
    st.session_state["show_intro"] = False
if st.sidebar.button("ℹ️ What is this bot?"):
    st.session_state["show_intro"] = not st.session_state["show_intro"]
if st.session_state["show_intro"]:
    st.sidebar.markdown("""
    ### Welcome to Wildlife Knowledge Assistant 🐾
    This bot lets you query wildlife data using Neo4j by selecting:
    - A county name
    - A reptile name
    Optionally, you can specify a task (e.g., Data Analysis or Conservation Management) to analyze the retrieved data.
    
    Enjoy exploring the wildlife knowledge base! 🌿
    """)

# 只保留 Advanced Selection 模式
st.markdown("### Please provide your selection")
county_name = st.text_input("Enter county name", key="county")
reptile_options = ["Select a reptile", "Alligator", "Turtle", "Lizard", "Snake"]
selected_reptile = st.selectbox("Select reptile name", reptile_options, key="reptile")
task_options = ["No further task", "Data Analysis", "Conservation Management"]
selected_task = st.selectbox("Select task (optional)", task_options, key="task")

if st.button("Submit Query", key="submit_query"):
    if selected_reptile == "Select a reptile" or county_name.strip() == "":
        st.write("Please select a valid reptile and enter a county name.")
    else:
        # 构造结构化查询字符串（仅用于展示信息）
        structured_query = f"Species: Reptile | Reptile Name: {selected_reptile} | County: {county_name}"
        if selected_task != "No further task":
            structured_query += f" | Task: {selected_task}"
        st.write("Structured Query:", structured_query)
        # 生成填空后的 Neo4j 查询
        neo4j_query = generate_query_from_template(selected_reptile, county_name)
        st.write("Generated Neo4j Query:", neo4j_query)
        # 执行查询
        kg_data = query_neo4j(neo4j_query)
        st.write("Query Output:", kg_data)
        # 如果有任务要求，则调用 LLM 进行数据分析
        if selected_task != "No further task":
            analysis_result = analyze_data_with_task(kg_data, selected_task, selected_reptile, county_name)
            st.write("Analysis Result:", analysis_result)
