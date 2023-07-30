import os

import chromadb
from chromadb.config import Settings

# load the root dir
ROOT_DIR=os.path.dirname(os.path.realpath(__file__))

# Define the folder path of source data
SOURCE_DIR=f"{ROOT_DIR}/data"

# Define folder path for storing database
PERSIST_DIR=f"{ROOT_DIR}/DB"

CHROMA_DB_SETTINGS=Settings(
    chroma_db_impl="duckdb+parquet",persist_directory=PERSIST_DIR,anonymized_telemetry=False)

# Embedding model path
EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

INGEST_THREADS=os.cpu_count or 8

# LLM model path
LLM_MODEL_NAME="llama-2-7b-chat.ggmlv3.q8_0.bin"

LLM_MODEL_PATH=f"{ROOT_DIR}/Models/llama2/{LLM_MODEL_NAME}"
