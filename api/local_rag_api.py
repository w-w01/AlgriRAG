from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# === 配置 ===
INDEX_FILE = "faiss_index/crop_structured_index.faiss"
DOCS_FILE = "faiss_index/crop_structured_docs.json"
MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# === 加载模型与数据 ===
index = faiss.read_index(INDEX_FILE)
with open(DOCS_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)

model = SentenceTransformer(MODEL_NAME)
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

# === /rag/by-label ===
@app.post("/rag/by-label")
async def rag_by_label(query: LabelQuery):
    crop = query.label.split("___")[0]
    disease = query.label.split("___")[1].replace("_", " ")
    query_text = f"{crop} - {disease}"

    retrieved = search_faiss(query_text, crop=crop, disease=disease)

    prompt = f"""
You are an agricultural assistant helping farmers diagnose crop diseases.

Identified disease: {disease} on {crop}.

Relevant documents:
- {retrieved[0]}
- {retrieved[1] if len(retrieved) > 1 else ''}
- {retrieved[2] if len(retrieved) > 2 else ''}

Please respond with:
1. Disease overview
2. Cause
3. Practical treatment advice
"""
    res = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })
    return {
        "answer": res.json().get("response", "[No response]"),
        "sources": retrieved
    }

# === /rag/by-text ===
@app.post("/rag/by-text")
async def rag_by_text(query: TextQuery):
    query_text = f"{query.crop} - unknown\nSymptom: {query.symptom}"
    retrieved = search_faiss(query_text, crop=query.crop)

    prompt = f"""
You are an agricultural assistant. A farmer growing {query.crop} described the following symptom:
"{query.symptom}"

Relevant documents:
- {retrieved[0]}
- {retrieved[1] if len(retrieved) > 1 else ''}
- {retrieved[2] if len(retrieved) > 2 else ''}

Please provide:
1. The most likely disease name
2. What causes it
3. Suggested field-level treatment
"""
    res = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })
    return {
        "answer": res.json().get("response", "[No response]"),
        "sources": retrieved
    }
