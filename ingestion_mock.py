"""
Mock Data Ingestion Script — ECSS Compliance Agent
This script processes the .txt mock files in the 'sample_data' folder
and builds the ChromaDB vector store.
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Configuration ─────────────────────────────────────────────────────────
DATA_DIR = "./sample_data"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

def build_mock_vector_store():
    print(f"[INFO] Looking for mock documents in '{DATA_DIR}'...")
    
    # 1. Load Text Documents (instead of PDFs)
    # We use DirectoryLoader to grab all .txt files in the sample_data folder
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        raise ValueError(f"[ERROR] No .txt documents found. Ensure mock data is in {DATA_DIR}.")
        
    print(f"[INFO] Successfully loaded {len(documents)} mock document(s).")

    # 2. Split text into manageable chunks
    print("[INFO] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[INFO] Created {len(chunks)} text chunks.")

    # 3. Generate Embeddings and build ChromaDB
    print(f"[INFO] Initializing embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"[INFO] Building and saving ChromaDB vector store to '{CHROMA_DB_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_DB_DIR
    )
    
    print("[SUCCESS] Mock vector store creation complete! Ready for testing.")

if __name__ == "__main__":
    build_mock_vector_store()
