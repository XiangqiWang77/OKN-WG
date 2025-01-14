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
    if not urls:
        st.write("No images available.")
        return

    cols = st.columns(len(urls))
    for i, url in enumerate(urls):
        try:
            # 设置 User-Agent 和 Referer
            headers = {
                "User-Agent": "MyImageFetcher/1.0 (https://wildlifelookup.streamlit.app/; xwang76@nd.edu)",
                "Referer": "https://www.wikimedia.org/"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # 检查 Content-Type
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                raise ValueError(f"URL does not point to an image: {url}")

            # 显示图片
            image = Image.open(BytesIO(response.content)).convert("RGB")
            cols[i].image(image, caption=f"Image {i+1}", use_column_width=True)
        except Exception as e:
            # 跳过无法加载的图片
            cols[i].write(f"Failed to load image {i+1}: {e}")



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
    options = ["Regular Response", "AI as Translator to KN-Wildlife", "AI as Toolbox for Aspect-Based Question", "AI as a Online(Wikidata) Searching Agent"]
    multimediaoptions = ["Text Only", "Text and Images"]

    selected_multimedia_mode = st.radio("Select output mode", multimediaoptions, horizontal=True)
    mode_selected = st.radio("Select external sources", options, horizontal=True)
    
    return [mode_selected, selected_multimedia_mode]

def extract_keywords_from_response(response_text):
    """
    Extracts important keywords or entities from the LLM-generated response using OpenAI GPT.
    """
    # Prompt for LLM to extract keywords
    prompt = f"""
    Extract the top 5 most relevant keywords or entities from the following text.
    Focus on nouns, names, or important terms relevant to the context.

    Text: {response_text}

    Return the keywords as a Python list of strings.
    """

    try:
        # Call the LLM API to process the prompt
        keywords_response = send_openai_prompt(prompt, model_name="gpt-4", temperature=0.5)

        # Ensure the result is a Python list (convert string output to Python object)
        keywords = eval(keywords_response.strip())
        return keywords

    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []  # Fallback to an empty list if extraction fails

def get_wikidata_id(entity_name):
    """
    Retrieves the Wikidata entity ID for a given entity name using Wikidata API.
    """
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
            # Return the first matching ID with the highest relevance
            return data['search'][0]['id']
    except Exception as e:
        print(f"Error fetching Wikidata ID for '{entity_name}': {e}")
    
    return None  # Return None if no entity ID is found

def validate_image_url(url, headers):
    """
    验证图片 URL 是否有效。
    返回 True 如果 URL 指向有效的图片文件，否则返回 False。
    """
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # 检查 Content-Type 是否以 image/ 开头
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("image/"):
            return True
    except Exception as e:
        print(f"Error validating image URL '{url}': {e}")
    return False


def search_images_from_keywords(keywords):
    """
    使用 Wikidata 的 SPARQL 查询端点，根据提取的关键词搜索图片。
    返回有效的图片 URL 列表，自动跳过加载失败的图片。
    """
    image_urls = []
    wikidata_sparql_endpoint = "https://query.wikidata.org/sparql"

    for keyword in keywords[:5]:
        # Step 1: 获取关键词的 Wikidata 实体 ID
        entity_id = get_wikidata_id(keyword)
        if not entity_id:
            continue  # 如果找不到实体 ID，跳过此关键词

        # Step 2: 构建 SPARQL 查询，查找实体的图片（P18 属性）
        query = f"""
        SELECT ?image WHERE {{
          wd:{entity_id} wdt:P18 ?image.
        }}
        LIMIT 1
        """

        # 设置 User-Agent 头部，确保符合 Wikidata 的用户代理策略
        headers = {
            "User-Agent": "MyImageFetcher/1.0 (https://wildlifelookup.streamlit.app/; xwang76@nd.edu)"
        }

        try:
            # 发送请求
            response = requests.get(
                wikidata_sparql_endpoint,
                params={"query": query, "format": "json"},
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            # 从返回的数据中提取图片 URL
            for result in data["results"]["bindings"]:
                if "image" in result:
                    image_url = result["image"]["value"]

                    # 测试图片 URL 是否有效
                    if validate_image_url(image_url, headers):
                        image_urls.append(image_url)
                    else:
                        print(f"Invalid image URL (skipped): {image_url}")

        except Exception as e:
            print(f"Error fetching image for keyword '{keyword}': {e}")

    return image_urls




# 处理不同模式下的逻辑
def handle_chat_mode(name, user_input):
    print(name[0])
    result = None

    if name[0] == "Regular Response":
        if name[1] == "Text Only":
            result = configure_llm_only_chain(user_input)
        elif name[1] == "Text and Images":
            # Step 1: Generate the textual response
            result = configure_llm_only_chain(user_input)

            # Step 2: Extract keywords/entities from the generated text
            keywords = extract_keywords_from_response(result)

            # Step 3: Search for images based on the extracted keywords
            image_urls = search_images_from_keywords(keywords)

            # Step 4: Display images if available
            if image_urls:
                display_images(image_urls)
            else:
                st.write("No images found for the extracted keywords.")

    elif name[0] == "AI as Translator to KN-Wildlife":
        translate_function = prompt_cypher(llm)  # 获取生成函数
        temp_result = translate_function(user_input)  # 调用生成函数获取 Cypher 查询

        print(temp_result)  # 检查 Cypher 查询是否正确生成

        if name[1] == "Text Only":
            rag_chain = configure_qa_rag_chain(
                llm,  # 第一个参数传递 llm
                query=temp_result,  # 传递生成的 Cypher 查询
                Graph_url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PASSWORD
            )
            result = rag_chain

        elif name[1] == "Text and Images":
            # 执行图谱查询
            kg_output = query_neo4j(temp_result)

            print("kg_output", kg_output)

            # 使用LLaVA输出处理多媒体数据
            feed_result = generate_llava_output(user_input, kg_output)
            result = feed_result["answer"]

            # 展示多媒体（图片等）
            keywords = extract_keywords_from_response(result)

            # Step 3: Search for images based on the extracted keywords
            image_urls = search_images_from_keywords(keywords)

            # Step 4: Display images if available
            if image_urls:
                display_images(image_urls)
            else:
                st.write("No images found for the extracted keywords.")

    elif name[0] == "AI as Toolbox for Aspect-Based Question":
        aspect_category = classfic(user_input, json_data, llm)
        temp_chain = generate_aspect_chain(
            question=user_input,
            multimedia_option=name[1],
            aspect=aspect_category,
            llm_name=llm,
            vllm_name=None
        )
        feed_result = temp_chain

        if name[1] == "Text Only":
            result = feed_result["answer"]
        elif name[1] == "Text and Images":
            result = feed_result["answer"]
            keywords = extract_keywords_from_response(result)

            # Step 3: Search for images based on the extracted keywords
            image_urls = search_images_from_keywords(keywords)

            # Step 4: Display images if available
            if image_urls:
                display_images(image_urls)
            else:
                st.write("No images found for the extracted keywords.")

    elif name[0] == "AI as a Online(Wikidata) Searching Agent":
        print("Function calling")

        # 通过 LLM 推断可能的节点范围
        node_scope = infer_node_scope_from_question(llm, user_input)

        # 执行 web search agent，传递节点范围限制
        agent_result = web_search_agent(
            llm=llm,
            question=user_input,
            node_scope=node_scope  # 传递节点范围
        )

        # 拼接最终查询
        Final_str = (
            f"question is {user_input} The web retrieval agent is {agent_result} "
            "Please provide an answer."
        )

        print(Final_str)

        if name[1] == "Text Only":
            result = send_openai_prompt(Final_str)
        elif name[1] == "Text and Images":
            result = send_openai_prompt(Final_str)
            # Add logic to process multimedia retrieval if required
            # Step 2: Extract keywords/entities from the generated text
            keywords = extract_keywords_from_response(result)

            # Step 3: Search for images based on the extracted keywords
            image_urls = search_images_from_keywords(keywords)

            # Step 4: Display images if available
            if image_urls:
                display_images(image_urls)
            else:
                st.write("No images found for the extracted keywords.")

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
