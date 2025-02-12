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

# 如有需要，可加载环境变量
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

# 利用 LLM 填充查询模板（只填空）
def generate_query_from_template(animal, county):
    # 根据动物类别确定节点标签
    animal_map = {
        "Reptile": "Reptile_name",
        "Amphibian": "Amphibian_name",
        "Fish": "Fish_name",
        "Bird": "Bird_name"
    }
    node_label = animal_map.get(animal, "Reptile_name")
    # Neo4j 查询模板，注意 Location 节点用来匹配县名
    query_template = (
        "MATCH (a:{node_label})-[:OBSERVED_AT]->(l:Location) "
        "WHERE toLower(l.name) CONTAINS toLower('{county}') "
        "RETURN a, l"
    )
    # 构造提示让 LLM 输出 JSON 对象填充占位符
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

# 若有任务要求，则利用 LLM 对查询到的数据进行分析
def analyze_data_with_task(kg_data, task, animal, county):
    prompt = f"""
Based on the following Neo4j query results: {kg_data},
please provide a detailed analysis for the wildlife management task "{task}".
The data is from animal category "{animal}" observed in a county containing "{county}".
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
st.write("使用选择模式查询动物数据，并可进行后续任务分析。")

if "show_intro" not in st.session_state:
    st.session_state["show_intro"] = False
if st.sidebar.button("ℹ️ What is this bot?"):
    st.session_state["show_intro"] = not st.session_state["show_intro"]
if st.session_state["show_intro"]:
    st.sidebar.markdown("""
    ### 欢迎使用 Wildlife Knowledge Assistant 🐾
    本工具允许您通过选择以下参数进行查询：
    - 输入 county 名称
    - 选择动物类别（Reptile、Amphibian、Fish、Bird）
    如有需要，可选择具体任务（例如 Data Analysis 或 Conservation Management），
    则系统将对查询到的数据进行进一步分析。
    
    请确保数据库中 Location 节点的 name 属性与输入的 county 匹配，
    以及相应动物节点（如 Reptile_name）中包含正确的动物名称。
    
    祝您使用愉快！
    """)

st.markdown("### 请选择查询条件")
county_name = st.text_input("请输入 county 名称", key="county")
animal_options = ["Select an animal", "Reptile", "Amphibian", "Fish", "Bird"]
selected_animal = st.selectbox("请选择动物类别", animal_options, key="animal")
task_options = ["No further task", "Data Analysis", "Conservation Management"]
selected_task = st.selectbox("请选择任务（可选）", task_options, key="task")

if st.button("提交查询", key="submit_query"):
    if selected_animal == "Select an animal" or county_name.strip() == "":
        st.write("请正确选择动物类别并输入 county 名称。")
    else:
        # 构造结构化查询字符串（仅用于展示）
        structured_query = f"Animal: {selected_animal} | County: {county_name}"
        if selected_task != "No further task":
            structured_query += f" | Task: {selected_task}"
        st.write("结构化查询：", structured_query)
        # 利用 LLM 填充模板生成 Neo4j 查询
        neo4j_query = generate_query_from_template(selected_animal, county_name)
        st.write("生成的 Neo4j 查询：", neo4j_query)
        # 执行查询
        kg_data = query_neo4j(neo4j_query)
        st.write("查询结果：", kg_data)
        # 如果有任务要求，则调用 LLM 对数据进行分析
        if selected_task != "No further task":
            analysis_result = analyze_data_with_task(kg_data, selected_task, selected_animal, county_name)
            st.write("分析结果：", analysis_result)
