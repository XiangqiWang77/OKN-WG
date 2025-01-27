def extract_keywords_from_question(question: str) -> dict:
    prompt = """
    You are an intelligent assistant specializing in querying a Neo4j database for animal observations in Florida. 
    Your task is to extract keywords from user questions to build Cypher queries.
    
    Note that animals specis are formatted like "Reptile_name" not "Reptile".

    Output Requirements:
    You must generate a dict formatted like this:

    {
    "specy_type": "Reptile_name",
    "location_attribute": "name",
    "location_value": "Alachua",
    "specy_attribute": "name",
    "specy_value": "",
    "observation_attribute": "dates",
    "observation_value": ""
    }

    Extract and return these fields as a plain string (not JSON) based on the given question.
    """

    # 使用 send_openai_prompt 直接调用 LLM
    response_text = send_openai_prompt(f"The question is: {question}\n{prompt}")

    # 解析字符串为字典
    try:
        keywords = json.loads(response_text)
    except json.JSONDecodeError:
        # 如果解析失败，返回默认值
        keywords = {
            "specy_type": "Animal_name",
            "location_attribute": "name",
            "location_value": "",
            "specy_attribute": "name",
            "specy_value": "",
            "observation_attribute": "dates",
            "observation_value": ""
        }
    return keywords



# 生成 Cypher 查询
def construct_cypher_query(kwargs, multi_option):
    # 从 kwargs 获取字段
    specy_type = kwargs.get("specy_type", "Animal_name")
    location_attribute = kwargs.get("location_attribute", "name")
    location_value = kwargs.get("location_value", "")
    specy_attribute = kwargs.get("specy_attribute", "name")
    specy_value = kwargs.get("specy_value", "")
    observation_attribute = kwargs.get("observation_attribute", "dates")
    observation_value = kwargs.get("observation_value", "")

    conditions = []
    # 根据字段构建 WHERE 条件
    if location_value:
        conditions.append(f'l.{location_attribute} CONTAINS "{location_value}"')
    if specy_value:
        conditions.append(f'r.{specy_attribute} CONTAINS "{specy_value}"')
    if observation_value:
        conditions.append(f'o.{observation_attribute} CONTAINS "{observation_value}"')

    # 生成最终 Cypher 查询
    where_clause = " AND ".join(conditions)
    
    cypher_query = f"""
    MATCH (r:{specy_type})-[o:OBSERVED_AT]->(l:Location)
    """
    if where_clause:
        cypher_query += f"WHERE {where_clause}\n"

    if multi_option == 1:
        cypher_query += """
        RETURN r.name AS specy_name, o.multimedia AS Multimedia, l.name AS location_name
        """
    else:
        cypher_query += """
        RETURN r, o, l
        """

    return cypher_query