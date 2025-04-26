import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# === 配置路径 ===
DATA_PATH = "sync/crop_disease_structured.json"
INDEX_DIR = "faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "crop_structured_index.faiss")
DOCS_FILE = os.path.join(INDEX_DIR, "crop_structured_docs.json")

# === 加载结构化数据 ===
with open(DATA_PATH, "r", encoding="utf-8") as f:
    entries = json.load(f)

# === 构建完整文段并嵌入 ===
model = SentenceTransformer("all-MiniLM-L6-v2")
docs = []
vectors = []

for entry in entries:
    text = f"{entry['crop']} - {entry['disease']}\n"
    text += f"Symptom: {entry.get('symptom', '')}\n"
    text += f"Cause: {entry.get('cause', '')}\n"
    text += f"Treatment: {entry.get('treatment', '')}"
    docs.append(text)
    vectors.append(model.encode(text))

vectors = np.array(vectors).astype("float32")

# === 构建并写入 FAISS 索引 ===
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

os.makedirs(INDEX_DIR, exist_ok=True)
faiss.write_index(index, INDEX_FILE)
with open(DOCS_FILE, "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=2)

print(f"✅ FAISS index built with {len(docs)} entries → {INDEX_FILE}")
