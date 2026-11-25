import asyncio
import json
import os
import time
from pathlib import Path

import pymysql
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
import sys

from pydantic import BaseModel, Field
from pymysql.cursors import DictCursor

ROOT_PATH = Path(__file__).parent.parent # 获取项目根目录
sys.path.insert(0,str(ROOT_PATH))
from utils.milvus_utils import get_milvus_client,get_embedding
from utils.geo_utils import direct,get_origin_location

load_dotenv()

agent = None

# 1、agent能够推荐特色菜
@tool
def search_feature_dish():
    """搜索餐厅的特色菜"""
    sql = """
        select
            menu_items.dish_name as 菜品名称,
            menu_items.price as 价格,
            menu_items.category as 菜品类别,
            menu_items.description as 描述,
            menu_items.spice_level as 麻辣程度,
            menu_items.flavor as 口味,
            menu_items.main_ingredients as 主料,
            menu_items.cooking_method as 烹饪方法,
            menu_items.is_vegetarian as 是否素食,
            menu_items.allergens as 过敏源
            from menu_items
        where is_featured=1

        """
    with pymysql.connect(
            host=os.getenv("MYSQL_HOST"),
            port=int(os.getenv("MYSQL_PORT")),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"),
            charset="utf8") as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            return result

# 2、根据用户需求推荐菜品
@tool
def user_favorite_dishes(query:str):
    """根据用户的口味要求推荐菜品"""
    client = get_milvus_client() # 获取milvus客户端
    embedding = get_embedding() # 获取嵌入模型

    vectory_query = embedding.embed_query(query)
    search_results = client.search(
        collection_name=os.getenv("MILVUS_COLLECTION"),
        data=[vectory_query],
        anns_field="vector",
        output_fields=["text"]
    )

    if search_results:
        all_results = search_results[0]
        final_results = []
        for result in all_results:
            if result["distance"] > 0.5:
                final_results.append(result["entity"]["text"])
        return final_results
    return "当前菜单中没有找到和用户喜好匹配的菜品！"

# 3、预定

class ReservationToolArgsInfo(BaseModel):
    num_people:int = Field(description="预订人数")
    num_children:int = Field(description="预订儿童人数")
    arrival_time:str = Field(description="到店时间，格式：YYYY-MM-DD HH")
    seat_preference:str= Field(description="座位偏好")
    main_dish_preference:str=Field(description="主菜偏好")
    other_comments:str= Field(description="其他备注")

@tool(args_schema=ReservationToolArgsInfo)
def make_reservation(num_people, num_children, arrival_time, seat_preference, main_dish_preference, other_comments):
    """餐厅座位约定"""
    try:
        sql = """
            insert into reservation_order (num_people, num_children, arrival_time, seat_preference, main_dish_preference, other_comments) values (%s, %s, %s, %s, %s, %s)
        """
        with pymysql.connect(
                host=os.getenv("MYSQL_HOST"),
                user=os.getenv("MYSQL_USER"),
                port=int(os.getenv("MYSQL_PORT")),
                password=os.getenv("MYSQL_PASSWORD"),
                database=os.getenv("MYSQL_DATABASE"),) as conn:
            with conn.cursor(DictCursor) as cursor:
                cursor.execute(sql, (num_people, num_children, arrival_time, seat_preference, main_dish_preference, other_comments))
                conn.commit() # 提交事务
                return "预订成功！"
    except Exception as e:
        return f"预订失败：{str(e)}"

# 4、提供导航路线
from langchain_mcp_adapters.client import MultiServerMCPClient

amap_tools = []
async def get_mcp_tools():
    global amap_tools
    if not amap_tools and len(amap_tools) == 0:
        client = MultiServerMCPClient(
            {
                "amap-maps": {
                    "transport": "streamable-http",
                    "url": "https://mcp.amap.com/mcp?key=08281e41a3e4030a5fd1d8ec2520a760"
                }
            }
        )
        amap_tools = await client.get_tools()
    return amap_tools

@tool
def is_in_range(destination_address:str):
    """根据用户提供的目的地，查询是否超出配送范围"""
    result = direct(address=destination_address)
    distance_km = result["distance_km"]
    return "在配送范围内" if distance_km <5.0 else "超出配送范围！"

async def get_agent():
    global agent
    if agent is None:
        amap_tools = await get_mcp_tools()
        all_tools = [search_feature_dish,user_favorite_dishes,make_reservation,is_in_range]+amap_tools
        checkpointer = InMemorySaver()
        # 获取系统提示词
        with open(ROOT_PATH / 'agent/prompt/system_prompt.txt', 'r', encoding='utf-8') as f:
            system_prompt = f.read()

        agent = create_agent(
            model = 'gpt-4o',
            tools = all_tools,
            system_prompt = system_prompt,
            checkpointer = checkpointer,
        )

    return agent

async def assistant_query(query:str, session_id:str):
    """将agent流式处理的结果，使用SSE流式返回给前端"""
    agent = await get_agent()
    config = {"configurable":{"thread_id":session_id}}
    current_data = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    async for chunk,metadata in agent.astream({
        "messages":
            [
                {
                    "role": "system",
                    "content": f"你是六合餐厅的智能助手，请根据用户的输入，进行回答，或者完成预订任务，或根据用户位置提供路线规划。当前时间：{current_data}，用户当前位置：上海人民广场"
                },
                {
                    "role": "user",
                    "content": query
                }]
    }, config, stream_mode="messages"):
        if type(chunk) == ToolMessage:
            continue
        # 构建符合SSE规范格式的数据 json字符串{"content":内容, "type":"token"}\n\n
        payload_str = json.dumps({"content": chunk.content, "type": "token"}, ensure_ascii=False)
        yield f"data: {payload_str}\n\n" # 符合SSE规范

async def test_agent():
    agent = await get_agent()
    config = {"configurable":{"thread_id":"1"}}
    current_data = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    result1 = await  agent.ainvoke({
        "messages":
            [
                {
                    "role":"system",
                    "content":f"你是六合餐厅的智能助手，请根据用户的输入，进行回答，或者完成预订任务，或根据用户位置提供导航路线。当前时间：{current_data}，用户当前位置：上海人民广场"
                },
                {
            "role": "user",
            "content": "请帮我约定一下明天下午4点的座位，总共4个人,其中有一位小朋友，位置最好靠窗，菜品主要偏辣一点，希望位置不要太吵闹！还有从人民广场要怎么去你们餐厅？"
        }]
    },config)
    print(result1["messages"][-1].content)

    res2 = await  agent.ainvoke({
        "messages":
            [
                {
                    "role": "user",
                    "content": "好的，以上信息没问题，请帮我预定！"
                }]
    }, config)
    print(res2["messages"][-1].content)

if __name__ == '__main__':
    asyncio.run(test_agent())
