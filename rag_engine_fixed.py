"""
ECSS Compliance RAG Engine — FIXED
────────────────────────────────────
Fix: HuggingFaceEndpoint con task="text-generation" non è più supportato
da Novita (provider default HF per Mistral) dalla versione huggingface_hub>=0.27.

Soluzione: usare ChatHuggingFace + HuggingFaceEndpoint con task="conversational"
che è l'approccio ufficiale moderno per modelli chat/instruct.

Compatibile con: langchain-huggingface>=0.0.3, huggingface_hub>=0.27
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DB_DIR   = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def format_docs(docs):
    """Aggregate retrieved document content into a unified context block."""
    return "\n\n".join(doc.page_content for doc in docs)


def initialize_rag_pipeline():
    """
    Initializes ChromaDB + HuggingFace chat LLM and assembles the LCEL RAG chain.
    
    KEY CHANGE vs previous version:
    - Removed task="text-generation"  →  replaced with ChatHuggingFace wrapper
    - ChatHuggingFace handles the [INST] formatting automatically for Mistral
    - Uses ChatPromptTemplate (system + human) instead of raw PromptTemplate
    """

    # ── 1. Vector store ────────────────────────────────────────────────────
    embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ── 2. LLM via HuggingFace Serverless API ─────────────────────────────
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError("[ERROR] HUGGINGFACEHUB_API_TOKEN not found. Check your .env file.")

    # HuggingFaceEndpoint — NO task parameter needed when wrapping with ChatHuggingFace
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_MODEL_ID,
        huggingfacehub_api_token=hf_token,
        max_new_tokens=512,
        temperature=0.1,
        repetition_penalty=1.15,
    )
    

    # ChatHuggingFace wraps the endpoint and handles chat formatting (incl. [INST] for Mistral)
    llm = ChatHuggingFace(llm=llm_endpoint, verbose=False)

    # ── 3. Prompt — ChatPromptTemplate (system + human roles) ─────────────
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert ESA AI compliance assistant specialised in ECSS 
(European Cooperation for Space Standardization) standards.
Your task is to help engineers verify that their designs comply with ECSS requirements.

CRITICAL INSTRUCTIONS:
1. If the answer is NOT in the provided context, say explicitly:
   "Confidence: LOW — The retrieved ECSS documents do not contain a definitive answer."
   Do NOT hallucinate. Never invent section numbers.
2. Always cite the specific ECSS standard and page number from the context.
3. Distinguish SHALL (mandatory), SHOULD (recommended) and MAY (optional).
4. Keep responses technical, precise, and actionable.

Retrieved ECSS context:
{context}"""
        ),
        ("human", "{question}"),
    ])

    # ── 4. LCEL RAG chain ─────────────────────────────────────────────────
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever
