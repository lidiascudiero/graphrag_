"""
Mock Data Ingestion Script — ECSS Compliance Agent
This script processes the .txt mock files in the 'sample_data' folder,
injects structural metadata (to prevent orphaned requirements),
and builds the ChromaDB vector store.
"""

import os
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
DATA_DIR = "./sample_data"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Regex to identify ECSS sections 
SECTION_PATTERN = re.compile(r"<?\b(\d+\.\d+(?:\.\d+)*)>?\s+([A-Z][^\n]{5,60})")

def build_mock_vector_store():
    print(f"[INFO] Looking for mock documents in '{DATA_DIR}'...")
    
    # 1. Load Text Documents (instead of PDFs)
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        raise ValueError(f"[ERROR] No .txt documents found. Ensure mock data is in {DATA_DIR}.")
        
    print(f"[INFO] Successfully loaded {len(documents)} mock document(s).")

    # 2. Split text into manageable chunks
    print("[INFO] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  
        chunk_overlap=200, 
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
   
    # Inject structural metadata to prevent orphaned chunks
 
    print("[INFO] Injecting normative structure metadata into chunks...")
    current_section = None
    current_source = None

    for chunk in chunks:
        # Reset section tracker if we switch to a new document
        doc_source = chunk.metadata.get("source")
        if doc_source != current_source:
            current_section = None
            current_source = doc_source

        # Look for section headers in the current chunk
        sections_found = list(SECTION_PATTERN.finditer(chunk.page_content))
        if sections_found:
            # Keep the last section found in this chunk as the active one
            last_match = sections_found[-1]
            sec_num = last_match.group(1)
            current_section = f"§{sec_num}"

        # Inject the active section into the metadata
        if current_section:
            chunk.metadata["section_id"] = current_section
            
    print(f"[INFO] Created {len(chunks)} text chunks with structural metadata.")

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