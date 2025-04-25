#!/bin/bash
uvicorn api.local_rag_api:app --host 0.0.0.0 --port 8000
