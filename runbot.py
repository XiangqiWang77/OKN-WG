import os
import streamlit as st
import json
from io import BytesIO
from PIL import Image
import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from toolbox.chains import configure_llm_only_chain, prompt_cypher, configure_qa_rag_chain, generate_llava_output, classfic, web_agent_multimedia
from toolbox.utils import ImageDownloader, convert_to_base64
from toolbox.aspects import generate_aspect_chain, extract_keywords_from_question
from toolbox.web_agent import web_search_agent, infer_node_scope_from_question 

# 加载环境变量
#load_dotenv(".env")

# Neo4j 数据库配置
NEO4J_URI="neo4j+s://f40686c2.databases.neo4j.io"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="RPW_MYabUDgsJzTrqDJLgDA2UzNrXC_rXYOLdP10tls"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# OpenAI 配置
api_key = st.secrets["OPENAI_KEY"]
client = OpenAI(api_key=api_key)

# 加载 aspect 分类的 JSON 数据
with open('aspects.json', 'r') as f:
    json_data = json.load(f)

# 调用 OpenAI 生成回答
def send_openai_prompt(prompt_text, model_name="gpt-4o", temperature=0.7):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text}],
            model=model_name,
            temperature=temperature,
            #max_tokens=token_limit
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Request failed: {e}"

# 自动补全问题
def get_autocomplete_suggestions(question):
    prompt = f"""
    You are given a graph containing information about animals and locations in Florida. Complete the user's partially entered question.
    Input: {question}
    """
    return send_openai_prompt(prompt)

# LLM加载
def load_llm(llm_name: str):
    return lambda prompt: send_openai_prompt(prompt, model_name=llm_name)

llm = load_llm("gpt-4o")

# 执行 Neo4j 查询
def query_neo4j(cypher_query, params=None):
    with driver.session() as session:
        result = session.run(cypher_query, params or {})
        return [record for record in result]

# 处理图像下载和展示
def display_images(urls):
    cols = st.columns(len(urls))
    for i, url in enumerate(urls):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        cols[i].image(image, caption=f"Image {i+1}", use_column_width=True)

# 处理聊天输入
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

# RAG模式选择
def mode_select() -> list:
    options = ["RAG Disabled", "LLM Translator RAG", "Aspects Matching RAG", "Wikidata Agent"]
    multimediaoptions = ["Multimedia Disabled", "Multimedia Enabled"]

    selected_multimedia_mode = st.radio("Select multimedia mode", multimediaoptions, horizontal=True)
    mode_selected = st.radio("Select RAG mode", options, horizontal=True)
    
    return [mode_selected, selected_multimedia_mode]

# 处理不同模式下的逻辑
def handle_chat_mode(name, user_input):

    print(name[0])
    result = None

    if name[0] == "RAG Disabled":
        output_function = configure_llm_only_chain(llm)
        result = output_function({"question": user_input})["answer"]

    elif name[0] == "LLM Translator RAG":
        translate_function = prompt_cypher(llm)  # 获取生成函数
        temp_result = translate_function(user_input)  # 调用生成函数获取 Cypher 查询

        print(temp_result)  # 检查 Cypher 查询是否正确生成

        if name[1] == "Multimedia Disabled":
            rag_chain = configure_qa_rag_chain(
                llm,  # 第一个参数传递 llm
                query=temp_result,  # 传递生成的 Cypher 查询
                Graph_url=NEO4J_URI, 
                username=NEO4J_USER, 
                password=NEO4J_PASSWORD
            )
            result = rag_chain

        elif name[1] == "Multimedia Enabled":
            # 执行图谱查询
            kg_output = query_neo4j(temp_result)

            print("kg_output",kg_output)

            # 使用LLaVA输出处理多媒体数据
            feed_result = generate_llava_output(user_input, kg_output)
            result = feed_result["answer"]

            # 展示多媒体（图片等）
            if "URLs" in feed_result and feed_result["URLs"]:
                display_images(feed_result["URLs"])


    elif name[0] == "Aspects Matching RAG":
        aspect_category = classfic(user_input, json_data, llm)
        #keywords = extract_keywords_from_question(user_input)
        temp_chain = generate_aspect_chain(
            question=user_input,
            multimedia_option=name[1],
            aspect=aspect_category,
            llm_name=llm,
            vllm_name=None
        )
        feed_result = temp_chain
        result = feed_result["answer"]
        if "URLs" in feed_result:
            display_images(feed_result["URLs"])



    elif name[0] == "Wikidata Agent":
        print("Function calling")
        
        # 通过 LLM 推断可能的节点范围
        node_scope = infer_node_scope_from_question(llm, user_input)

        #print("Node scope:", node_scope)
        
        # 执行 web search agent，传递节点范围限制
        agent_result = web_search_agent(
            llm=llm,
            question=user_input,
            node_scope=node_scope  # 传递节点范围
        )
        
        # 打印返回结果
        #print(agent_result)

        Final_str="question is"+str(user_input)+"The web retreival agent is"+str(agent_result)+"Please provide an answer."
        
        print(Final_str)


        result=send_openai_prompt(Final_str)

        print(result)
        # 使用爬取结果进行多媒体检索


    # 增加兜底机制
    if result is None:
        result = "No result generated. Please refine your question or try a different mode."

    return result


# 页面初始化
st.title("Wildlife Knowledge Assistant")

name = mode_select()
user_input_text = st.text_input("What would you like to know?")

if user_input_text:
    result = handle_chat_mode(name, user_input_text)
    st.write(result)
