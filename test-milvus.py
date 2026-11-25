# 学习如何使用向量数据库
from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv
import os
load_dotenv()

# 创建向量数据库
client = MilvusClient(
    url=os.getenv("MILVUS_URL"),
    token=os.getenv("MILVUS_TOKEN"),
)
collection_name = os.getenv("MILVUS_COLLECTION") #定义集合名称
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

# 确定集合中的字段及类型
schema=MilvusClient.create_schema(auto_id=True)  #保持自增
schema.add_field(field_name='id',datatype=DataType.INT64,is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=256)
# 定义索引类型
index_params = MilvusClient.prepare_index_params() #索引参数容器
index_params.add_index(
    field_name='vector', #为 vector 字段准备
    index_type='HNSW', #算法
    metric_type='IP',    #内积
)

client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params
)

# 2. 添加数据

text = [
    "hello world",
    "how are you?",
    "I am fine, thank you",
    "how old are you ?"
]
# 2.1 生成向量
embedding = HuggingFaceEmbeddings(
    model_name="./model/bge-base-zh",
    model_kwargs={
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    encode_kwargs={"normalize_embeddings": True} # 模型进行归一化， 配合IP
)

vector_list = embedding.embed_documents(text)
data_to_insert = [
    {"vector":vec, "text":txt } for vec, txt in zip(vector_list, text)
]

client.insert(
     collection_name=collection_name,
     data=data_to_insert
)


# 3. 查询数据