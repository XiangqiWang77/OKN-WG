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

# 加载环境变量（如果需要）
# load_dotenv()

# OpenAI 配置
api_key = st.secrets["OPENAI_KEY"]
client = OpenAI(api_key=api_key)

# Neo4j 数据库配置
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# 发送请求到 OpenAI
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

# 解析用户问题，提取关键词（保持原有 prompt 不变）
def extract_keywords_from_question(question: str) -> dict:
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
    response_text = send_openai_prompt(prompt)
    try:
        categories = json.loads(response_text)
    except json.JSONDecodeError:
        categories = {
            "source_node_category": "Reptile_name",
            "target_node_category": "Location"
        }
    return categories

# 构造 Cypher 查询（保持原有逻辑，但修改 county 匹配为模糊匹配，提升健壮性）
def construct_cypher_query(kwargs, multi_option) -> str:
    source_node_category = kwargs.get("source_node_category", "Reptile_name")
    target_node_category = kwargs.get("target_node_category", "Location")
    cypher_query = f"""
    MATCH (s:{source_node_category})-[r:OBSERVED_AT]->(t:{target_node_category})
    WHERE toLower(t.name) CONTAINS toLower('{kwargs.get('county', '')}')
    """
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

# 处理查询和 LLM 交互，生成最终答案（用于 free text 调用 generate_aspect_chain）
def generate_aspect_chain(question, multimedia_option, llm_name, aspect=None, vllm_name=None):
    kwargs = extract_keywords_from_question(question)
    print(kwargs)
    formatted_query = construct_cypher_query(kwargs, multi_option=0)
    print("formatted_query", formatted_query)
    kg_output = query_neo4j(formatted_query)
    kg_output = kg_output[:10000]
    prompt = f"""
    Use the following knowledge graph output to answer the user's question:
    - Question: {question}
    - Aspect: {aspect}
    - KG Output: {kg_output}
    """
    response = send_openai_prompt(prompt)
    print(response)
    print("kg_output", kg_output)
    return {"answer": response}

# 生成问答任务（保持不变）
def generate_ticket(title, body):
    prompt = f"""
    Generate a new question for the knowledge graph based on the following example:

    Title: {title}
    Body: {body[:200]}

    Create a similar question in the same style.
    """
    result = send_openai_prompt(prompt)
    return result

# 新增：构造结构化查询专用的 Cypher 查询函数，直接基于 species 与 county 构造查询（使用模糊匹配）
def construct_selection_cypher_query(species, county, multi_option=0):
    species_map = {
       "Fish": "Fish_name",
       "Reptile": "Reptile_name",
       "Amphibian": "Amphibian_name",
       "Birds": "Bird_name"
    }
    node_label = species_map.get(species, "Reptile_name")
    # county 使用模糊匹配，避免因数据中存储全称而查不到
    if multi_option == 1:
        cypher_query = f'''
        MATCH (s:{node_label})-[:OBSERVED_AT]->(c:Location)
        WHERE toLower(c.name) CONTAINS toLower("{county}")
        RETURN s.name AS species_name, s.multimedia AS multimedia, c.name AS county_name
        '''
    else:
        cypher_query = f'''
        MATCH (s:{node_label})-[:OBSERVED_AT]->(c:Location)
        WHERE toLower(c.name) CONTAINS toLower("{county}")
        RETURN s, c
        '''
    return cypher_query

# 处理不同模式下的逻辑
def handle_chat_mode(mode_info, user_input):
    print(mode_info[0])
    result = None
    # 若检测到结构化查询（Advanced Selection Mode），要求输入格式为："Species: X | County: Y | Task: Z"
    if "Task:" in user_input:
        parts = [part.strip() for part in user_input.split("|")]
        if len(parts) >= 3:
            species_value = parts[0].split(":", 1)[1].strip() if ":" in parts[0] else ""
            county_value = parts[1].split(":", 1)[1].strip() if ":" in parts[1] else ""
            task_value = parts[2].split(":", 1)[1].strip() if ":" in parts[2] else ""
        else:
            species_value = county_value = task_value = ""
        # 构造结构化查询专用的 Cypher 查询（只使用 species 与 county）
        cypher_query = construct_selection_cypher_query(species_value, county_value, multi_option=0)
        kg_data = query_neo4j(cypher_query)
        # 若任务为 Data Display，则直接展示查询到的数据
        if "data display" in task_value.lower():
            result = f"Data Retrieved: {kg_data}"
        else:
            prompt = (
                f"Based on the retrieved data: {kg_data}, please provide a detailed answer "
                f"for the following wildlife management task: {task_value}. "
                f"Data comes from species: {species_value} in county: {county_value}."
            )
            result = send_openai_prompt(prompt)
        return result
    else:
        # 非结构化查询按照原有逻辑处理
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
            Final_str = (
                f"Question: {user_input}. The web retrieval agent returns: {agent_result}. "
                "Please provide an answer."
            )
            print(Final_str)
            if mode_info[1] == "Text Only":
                result = send_openai_prompt(Final_str)
            elif mode_info[1] == "Text and Images":
                result = send_openai_prompt(Final_str)
                keywords = extract_keywords_from_response(result)
                image_urls = search_images_from_keywords(keywords)
                if image_urls:
                    display_images(image_urls)
                else:
                    st.write("No images found for the extracted keywords.")
        if result is None:
            result = "No result generated. Please refine your question or try a different mode."
        return result

# 保活线程相关代码
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
        import time
        time.sleep(interval)

def keep_streamlit_alive():
    while True:
        import time
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

# 处理 URL 查询参数
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
    - Ask complex questions and receive detailed answers powered by **LLM**.
    - Explore multimedia (text and images) information related to your questions.
    - Discover more about wildlife in the United States and beyond.

    **Features**:
    - **Interactive Modes**: Choose from various response modes (text-only, multimedia, etc.).
    - **Custom AI Models**: Powered by GPT-based LLMs and integrated APIs for a rich experience.
    - **Neo4j Database**: Provides real-time data querying and visualization.

    **How to use**:
    1. Type your question in the input box.
    2. Select a mode and hit submit.
    3. Explore the detailed results with optional images.

    Enjoy exploring the wildlife knowledge base! 🌿
    """)

st.markdown("### Choose your input mode")
input_mode = st.radio("Select input mode", options=["Free Text", "Advanced Selection Mode"], key="input_mode")
mode_info = mode_select()

if query:
    st.write(f"**Query from URL:** {query}")
    result = handle_chat_mode(mode_info, query)
    st.write(result)
else:
    if input_mode == "Advanced Selection Mode":
        # Advanced Selection Mode：选择 species、输入 county、选择任务
        species_options = ["Select a species", "Fish", "Reptile", "Amphibian", "Birds"]
        selected_species = st.selectbox("Select species", species_options, key="species")
        county_name = st.text_input("Enter county name", key="county")
        target_options = ["Select a task", "Data Display", "Data Analysis", "Conservation Management"]
        selected_target = st.selectbox("Select task", target_options, key="target")
        if st.button("Submit Advanced Query", key="submit_advanced"):
            query_parts = []
            if selected_species != "Select a species":
                query_parts.append(f"Species: {selected_species}")
            if county_name:
                query_parts.append(f"County: {county_name}")
            if selected_target != "Select a task":
                query_parts.append(f"Task: {selected_target}")
            structured_query = " | ".join(query_parts)
            result = handle_chat_mode(mode_info, structured_query)
            st.write(result)
    else:
        # Free Text Mode
        user_input_text = st.text_input("What would you like to know?", key="user_input_key")
        if user_input_text:
            result = handle_chat_mode(mode_info, user_input_text)
            st.write(result)
