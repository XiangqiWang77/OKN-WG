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

# 加载环境变量
#load_dotenv()

# Neo4j 配置
NEO4J_URI="neo4j+s://f40686c2.databases.neo4j.io"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="RPW_MYabUDgsJzTrqDJLgDA2UzNrXC_rXYOLdP10tls"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# 1. 推断节点范围
def infer_node_scope_from_question(llm, question):
    prompt = f"""
    Based on the following question, infer the most relevant node types from Neo4j to query. 
    Node types include: 'Amphibian_name', 'Bird_name', 'Reptile_name', 'Fish_name', 'Location'.
    
    Question: {question}

    Return the final result only as plain text, like 'Reptile_name'. No explanation, no bullshit.
    """
    response = send_openai_prompt(prompt)
    return response.strip()


# 2. 构建Cypher查询
def construct_cypher_query_with_scope(node_scope):
    cypher_query = f"""
    MATCH (n)
    WHERE n.wikidata_id IS NOT NULL 
      AND n:{node_scope}
    RETURN n.wikidata_id AS wikidata_id
    LIMIT 5
    """
    return cypher_query


# 3. 执行Neo4j查询，获取Wikidata ID
def fetch_wikidata_ids_from_neo4j(node_scope):
    query = """
    MATCH (n)
    WHERE n.wikidata_id IS NOT NULL 
      AND n:{node_scope}
    RETURN n.wikidata_id AS wikidata_id
    LIMIT 2
    """.format(node_scope=node_scope)  # 插入节点范围

    with driver.session() as session:
        result = session.run(query)
        return [record['wikidata_id'] for record in result]



import requests
from bs4 import BeautifulSoup
import time


# 4. 爬取Wikidata网页内容
def retrieve_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # 如果返回状态码不是200，将触发异常
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve content from {url}: {e}")
        return None


# 5. 判断答案是否存在
def judgement(question, content):
    for text in content:
        if question.lower() in text.lower():
            return f"Based on the content, the answer to '{question}' is found."
    return "No"


# 6. 提取更多相关链接
def extract_top_k_links(weburl, topK):
    try:
        response = requests.get(weburl, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)

        # 过滤出以 http 开头的链接，避免无效链接
        relevant_links = [link['href'] for link in links if link['href'].startswith('http')][:topK]
        return relevant_links
    except requests.exceptions.RequestException as e:
        print(f"Failed to extract links from {weburl}: {e}")
        return []


# 7. Web搜索代理主函数
def web_search_agent(llm, question, node_scope=""):
    # Step 1: 从 Neo4j 提取 Wikidata ID，限制节点范围
    wikidata_ids = fetch_wikidata_ids_from_neo4j(node_scope)
    if not wikidata_ids:
        return "No relevant Wikidata IDs found in the knowledge graph."

    # Step 2: 构建 Wikidata 链接
    wikidata_links = [f"https://www.wikidata.org/wiki/{id}" for id in wikidata_ids]

    # Step 3: 爬取页面内容
    all_content = []
    for link in wikidata_links:
        print(f"访问中: {link}")
        content = retrieve_content(link)
        if content:
            all_content.append(content)

    print(all_content)
    return all_content
