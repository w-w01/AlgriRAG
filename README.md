# 🧠 CropCare RAG System

This folder contains the **Retrieval-Augmented Generation (RAG)** pipeline for [CropCare](https://github.com/w-w01/AlgriRAG), an AI-powered assistant for crop disease detection and management. It combines **image-based disease classification** with **AI-generated disease insights**, enabling farmers to take informed action faster.

## 📦 Structure

```
api/
├── main.py                          # FastAPI entry point for inference
├── requirements.txt                # Python dependencies
├── faiss_index/
│   ├── crop_structured_docs.json   # Documents used for RAG (JSON format)
│   └── crop_structured_index.faiss # FAISS index built from the documents
├── sync/
│   ├── crop_disease_structured.json  # Raw disease info before indexing
│   └── generate_structured_faiss.py  # Script to generate FAISS index
```

## ⚙️ What It Does

- Retrieves related disease documents using a **FAISS-based vector database**.
- Uses **Azure OpenAI** to generate helpful, context-aware answers from the retrieved documents.

## 🚀 How to Run Locally

1. **Install dependencies**

```bash
cd api
pip install -r requirements.txt
```

2. **Generate FAISS index**

```bash
python sync/generate_structured_faiss.py
```

3. **Run the API**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or use the startup script:

```bash
bash startup.sh
```

## ☁️ Cloud Deployment

This project is designed to be deployed on **Azure App Service**. On boot, the API fetches the FAISS index and document JSON from Azure Blob Storage and loads them into memory.

Make sure to configure:

- `AZURE_BLOB_CONNECTION_STRING`
- `AZURE_CONTAINER_NAME`
- `AZURE_OPENAI_KEY`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_ENDPOINT`

All of the above should be injected via `.env` or App Service configuration.

## 📄 API Endpoint

```
POST /rag/by-label
```

**Request body:**

```json
{
  "label": "Tomato___Late_blight"
}
```

**Response:**

```json
{
    "answer": "### 1. Disease Overview\nLate blight is a serious disease affecting tomato plants, characterized by the development of large, irregularly shaped, water-soaked lesions on leaves, stems, and fruit. These lesions often have a greasy appearance and can expand rapidly, particularly in cool and moist weather conditions. Additionally, white mold may be visible on the undersides of affected leaves, indicating the presence of the pathogen.\n\n### 2. Cause\nThe disease is caused by the oomycete pathogen **Phytophthora infestans**, which thrives in cool, humid environments. It spreads quickly through rain splash and overhead irrigation, making it crucial for farmers to monitor their crops, especially during periods of wet weather.\n\n### 3. Practical Treatment Advice\nTo manage late blight in tomato plants, follow these treatment recommendations:\n\n- **Preventive Fungicide Application**: Apply protective fungicides, such as **chlorothalonil** or **copper hydroxide**, proactively before visible symptoms appear. This is essential for controlling the spread of the disease.\n  \n- **Cultural Practices**: Implement good cultural practices, such as crop rotation and selecting resistant tomato varieties, to reduce the risk of infection.\n\n- **Water Management**: Avoid overhead irrigation and instead use drip irrigation to minimize moisture on the foliage, which can help reduce the risk of late blight.\n\n- **Regular Monitoring**: Regularly inspect your plants for early signs of disease and take immediate action if symptoms are observed.\n\n- **Remove Infected Plant Material**: Promptly remove and destroy any infected plant material to prevent the spread of the pathogen to healthy plants.\n\nBy following these guidelines, you can effectively manage late blight and protect your tomato crops.",
    "sources": [
        "tomato - Late Blight\nSymptom: Large, irregularly shaped, water-soaked lesions form on leaves, stems, and fruit. Lesions may appear greasy and rapidly expand during cool, moist weather. White mold may be seen on leaf undersides.\nCause: Caused by the oomycete pathogen Phytophthora infestans. The disease spreads quickly in cool, humid environments, especially with overhead irrigation or rain splash.\nTreatment: Apply protective fungicides such as chlorothalonil or copper hydroxide before symptoms appear. Remove and destroy infected plants immediately, and avoid planting near potatoes."
    ]
}
```

```
POST /rag/by-text
```

**Request body:**

```json
{
  "crop": "tomato",
  "symptom": "Small, circular spots with dark edges and gray centers appear on lower tomato leaves, leading to yellowing, wilting, and leaf drop."
}
```

**Response:**

```json
{
    "answer": "1. The most likely disease name: **Target Spot** (caused by Corynespora cassiicola)\n\n2. What causes it: The disease is caused by the fungus **Corynespora cassiicola**. It thrives in warm, humid conditions and spreads through spores via wind, water splash, or contaminated tools.\n\n3. Suggested field-level treatment: \n   - Apply fungicides such as **chlorothalonil** or **strobilurins** preventively to manage the disease.\n   - Additionally, remove and properly dispose of infected plant material to reduce the spread of the fungus.",
    "sources": [
        "tomato - Target Spot\nSymptom: Circular brown lesions with concentric rings appear on leaves, often surrounded by yellow halos. Lesions may coalesce, causing leaf blight and premature defoliation. Fruits can also be affected in severe cases.\nCause: Caused by the fungus Corynespora cassiicola. The disease is favored by warm, humid conditions and spreads via spores in wind, water splash, or contaminated tools.\nTreatment: Apply fungicides such as chlorothalonil or strobilurins preventatively. Remove infected leaves and debris, improve air circulation, and avoid overhead irrigation.",
        "tomato - Septoria Leaf Spot\nSymptom: Numerous small, circular spots with dark brown margins and grayish-white centers appear on older leaves, typically starting from the bottom of the plant. Infected leaves may yellow and drop prematurely.\nCause: Caused by the fungus Septoria lycopersici, which overwinters in plant debris and soil. The disease is spread by splashing water, tools, and workers, especially in wet, humid conditions.\nTreatment: Remove infected leaves and debris. Avoid overhead watering and ensure proper plant spacing. Apply fungicides like chlorothalonil, mancozeb, or copper-based formulations as a preventative measure.",
        "tomato - Early Blight\nSymptom: Older leaves develop dark brown to black lesions with concentric rings, forming a target-like pattern. Affected leaves often turn yellow and drop prematurely, starting from the bottom of the plant.\nCause: Caused by the fungus Alternaria solani, which overwinters in soil and plant debris. It thrives in warm, wet conditions and spreads via wind, rain splash, and mechanical means.\nTreatment: Use resistant tomato varieties and remove infected leaves promptly. Apply fungicides like mancozeb or azoxystrobin at regular intervals and maintain good air circulation through proper spacing."
    ]
}
```



## 🧪 Index Rebuilding

To update disease info:

1. Edit `sync/crop_disease_structured.json`
2. Re-run `generate_structured_faiss.py`
