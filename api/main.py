from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import json
from sentence_transformers import SentenceTransformer
import os
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI

required_envs = [
    "AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_CONTAINER_NAME",
    "AZURE_BLOB_FAISS_INDEX", "AZURE_BLOB_DOCS",
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION"
]

for var in required_envs:
    if var not in os.environ:
        raise EnvironmentError(f"Missing environment variable: {var}")

# 环境变量中读取
AZURE_STORAGE_CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
BLOB_CONTAINER = os.environ["AZURE_STORAGE_CONTAINER_NAME"]
BLOB_INDEX_FILE = os.environ["AZURE_BLOB_FAISS_INDEX"]
BLOB_DOC_FILE = os.environ["AZURE_BLOB_DOCS"]

blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONN_STR)
container_client = blob_service.get_container_client(BLOB_CONTAINER)

def download_blob_to_local(blob_name, local_path):
    blob_client = container_client.get_blob_client(blob=blob_name)
    with open(local_path, "wb") as f:
        data = blob_client.download_blob()
        f.write(data.readall())

# === 配置 ===#
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
)

DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
TOKEN_LIMIT = int(os.environ.get("AZURE_OPENAI_TOKEN_LIMIT", "120000"))

# === 加载模型与数据 ===
LOCAL_INDEX_PATH = "faiss_index/crop_structured_index.faiss"
LOCAL_DOC_PATH = "faiss_index/crop_structured_docs.json"

download_blob_to_local(BLOB_INDEX_FILE, LOCAL_INDEX_PATH)
download_blob_to_local(BLOB_DOC_FILE, LOCAL_DOC_PATH)

if not os.path.exists(LOCAL_INDEX_PATH) or not os.path.exists(LOCAL_DOC_PATH):
    raise RuntimeError("Failed to load FAISS index or documents from blob storage.")

index = faiss.read_index(LOCAL_INDEX_PATH)
with open(LOCAL_DOC_PATH, "r", encoding="utf-8") as f:
    docs = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
app = FastAPI()

# === 请求模型 ===
class LabelQuery(BaseModel):
    label: str  # e.g. Tomato___Late_blight

class TextQuery(BaseModel):
    crop: str   # e.g. tomato
    symptom: str  # e.g. Leaves turn yellow with small dark spots

# === 查询构造工具 ===
def search_faiss(query_text: str, crop: str = None, disease: str = None):
    vector = model.encode([query_text]).astype("float32")
    D, I = index.search(vector, k=6)
    retrieved_all = [docs[i] for i in I[0]]

    if crop or disease:
        crop = crop.lower() if crop else ""
        disease = disease.lower() if disease else ""
        filtered = [
            doc for doc in retrieved_all
            if crop in doc.lower() and disease in doc.lower()
        ]
        return filtered[:3] if filtered else retrieved_all[:3]
    else:
        return retrieved_all[:3]

@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.post("/rag/by-label")
async def rag_by_label(query: LabelQuery):
    crop = query.label.split("___")[0]
    disease = query.label.split("___")[1].replace("_", " ")
    query_text = f"{crop} - {disease}"

    retrieved = search_faiss(query_text, crop=crop, disease=disease)
    # to avoid exceeding token limit
    retrieved_doc = retrieved[0][:500]
    prompt = f"""
You are an agricultural assistant helping farmers diagnose crop diseases.

Identified disease: {disease} on {crop}.

Relevant documents:
- {retrieved_doc}

Please respond with:
1. Disease overview
2. Cause
3. Practical treatment advice
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are an agricultural assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048,
        temperature=0.7
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": retrieved
    }

@app.post("/rag/by-text")
async def rag_by_text(query: TextQuery):
    query_text = f"{query.crop} - unknown\nSymptom: {query.symptom}"
    retrieved = search_faiss(query_text, crop=query.crop)
    # to avoid exceeding token limit
    retrieved_doc = retrieved[0][:500]
    prompt = f"""
You are an agricultural assistant. A farmer growing {query.crop} described the following symptom:
"{query.symptom}"

Relevant documents:
- {retrieved_doc}

Please provide:
1. The most likely disease name
2. What causes it
3. Suggested field-level treatment
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are an agricultural assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048,
        temperature=0.7
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": retrieved
    }
