# 将后端能力抽象成具体接口，供前端调用
import json
import os
from typing import Optional, List

import httpx
import pymysql
from fastapi import FastAPI
from langchain_core.outputs import ChatResult
from pydantic import BaseModel, Field
from pymysql.cursors import DictCursor
from starlette.responses import StreamingResponse
from datetime import datetime
from agent.langchain_assistant import assistant_query,get_agent
from utils.geo_utils import bike, drive, direct
from utils.redis_utils import get_redis_client

app = FastAPI()

class FAQItem(BaseModel):
    id: str
    question: str
    answer: str

class ChatRequest(BaseModel):
    query: str = Field(description="用户输入的聊天内容")

class ReservationItem(BaseModel):
    id: int
    num_people: int
    num_children: int
    arrival_time: Optional[str] = None # 非必填项
    seat_preference: Optional[str] = None
    main_dish_preference: Optional[str] = None
    other_comments: Optional[str] = None
    created_at: Optional[str] = None

class ReservationListResponse(BaseModel):
    success: bool
    reservations: list[ReservationItem]
    count: int
    message: str

class MenuItem(BaseModel):
    id:int
    dish_name:str
    price:float
    category:Optional[str] = None
    description:Optional[str] = None
    spice_level:Optional[int] = None
    flavor:Optional[str] = None
    main_ingredients : Optional[str] = None
    cooking_method:Optional[str] = None
    is_vegetarian : Optional[int] = None
    allergens:Optional[str] = None
    formatted_price: Optional[str] = None
    spice_text: Optional[str] = None

class MenuListResponse(BaseModel):
    success: bool
    menu_items: list[MenuItem]
    count: int
    message: str

class DeliveryResponse(BaseModel):
    success: bool
    message: str
    in_range:Optional[bool] = True
    address:Optional[str] = None
    resolved_address:Optional[str] = None
    distance_km:Optional[float] = None
    duration_min:Optional[float] = None

class DeliveryRequest(BaseModel):
    address: str
    travel_mode: Optional[str] = '2'

class FAQResponse(BaseModel):
    success: bool
    query: str
    suggestions: List[FAQItem]

def load_faq_from_redis()->list[FAQItem]:
    client = get_redis_client()
    faq_all_ids = client.smembers('faq:ids')
    all_keys = [f"faq:item:{id}" for id in faq_all_ids]
    pipeline = client.pipeline()
    for key in all_keys:
        pipeline.hgetall(key)
    faq_items = pipeline.execute()
    return [FAQItem(**faq_item) for faq_item in faq_items]

# 流式返回的聊天接口
@app.post("/chat")
def chat_endpoint(request:ChatRequest):
    """提供一个聊天接口"""
    # 流式返回
    session_id = "session_001"
    return StreamingResponse(assistant_query(query=request.query, session_id=session_id),media_type="text/event-stream")

# 预定接口
@app.get("/reservation/list", response_model=ReservationListResponse)
def book_endpoint(limit:int = 1): # 暂时不用limit
    sql = """
        select
             id,
             num_people,
             num_children,
             arrival_time,
             seat_preference,
             main_dish_preference,
             other_comments,
             created_at
        from reservation_order

        """

    with pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        charset="utf8",
    ) as conn:
        with conn.cursor(DictCursor) as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()
            items = []
            for item in results:
                items.append(ReservationItem(
                    id=item["id"],
                    num_people=item["num_people"],
                    num_children=item["num_children"],
                    arrival_time=datetime.strftime(item["arrival_time"], "%Y-%m-%d %H:%M:%S"),
                    seat_preference=item["seat_preference"],
                    main_dish_preference=item["main_dish_preference"],
                    other_comments=item["other_comments"],
                    created_at=datetime.strftime(item["created_at"], "%Y-%m-%d %H:%M:%S"),
                ))
            return ReservationListResponse(success=True,reservations=items,count=len(items),message="成功")

# 菜单列表查询接口
@app.get("/menu/list",response_model=MenuListResponse)
def menu_list(limit:int=10):
    sql = f"""
    select 
        id,
        dish_name,
        price,
        category,
        description,
        spice_level,
        flavor,
        main_ingredients,
        cooking_method,
        is_vegetarian,
        allergens,
        concat(price,'元') as formatted_price,
        case when spice_level=0 then '不辣'
             when spice_level=1 then '微辣'
             when spice_level=2 then '中辣'
             when spice_level=3 then '重辣'
        else '无辣'
        end as spice_text
    from menu_items
    limit {limit}
    """

    with pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        charset="utf8",
    ) as conn:
        with conn.cursor(DictCursor) as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()
            items = []
            for item in results:
                items.append(MenuItem(**item))
            return MenuListResponse(success=True,menu_items=items,count=len(items),message='成功')

# 配送接口
@app.post("/delivery",response_model=DeliveryResponse)
def delivery(request:DeliveryRequest):
    try:
        address = request.address
        travel_mode = request.travel_mode
        func={
            '1':bike,
            '2':drive,
            '3':direct,
        }
        result =  func[travel_mode](address)
        distance_km = result["distance_km"]
        in_range = True if distance_km < 5.0 else False
        return DeliveryResponse(
            success=True,
            message="在配送范围内" if in_range else "超出配送范围！",
            in_range=in_range,
            address=address,
            resolved_address=result["resolved_address"],
            distance_km=distance_km,
            duration_min=result["duration_min"],
        )
    except Exception as e:
        return DeliveryResponse(success=False,message=str(e))

from difflib import SequenceMatcher
import re
# 定义正则表达式，用于移除文本中的标点符号
_PUNCT_RE = re.compile(r"[\s,，。.!！？?;；:：'\"“”‘’（）()\[\]【】<>《》、/\\|_\-]+")
# 对文本进行清洗的函数：
def _normalize_text(text: str) -> str:
    return _PUNCT_RE.sub("", (text or "").strip())

def get_similarity_score(query:str, faq_question:str):
    query = _normalize_text(query)
    faq_question = _normalize_text(faq_question)

    # 1、计算两个字符串的相似度得分
    similarity_score = SequenceMatcher(None, query, faq_question).ratio()
    # 2、计算两个字符串的jaccard相似度得分：交集长度/并集长度
    query = set(list(query))
    faq_question = set(list(faq_question))
    jaccard_score = len(query.intersection(faq_question)) / len(query.union(faq_question))
    # 3、计算两个字符串的包含关系得分
    bonus = 0
    if query in faq_question:
        bonus = 0.5
    return similarity_score * 0.6 + bonus + jaccard_score

@app.get("/faq/suggest", response_model=FAQResponse)
def faq_suggest_endpoint(query:str, limit:int=10):
    # 1、从redis中读取所有FAQ
    faq_items = load_faq_from_redis()

    # 2、根据用户问题进行匹配
    scores_list = []
    for faq_item in faq_items:
        score = get_similarity_score(query, faq_item.question)
        scores_list.append((score,faq_item))

    # 3、根据得分排序，取前limit个FAQ项
    scores_list.sort(key=lambda x:x[0], reverse=True)
    limit=2
    top_k_faq_items = [faq_item for _,faq_item in scores_list[:limit]]

    # 4、返回
    return FAQResponse(
        success=True,
        query=query,
        suggestions=top_k_faq_items,
    )
