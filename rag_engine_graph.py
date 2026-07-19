"""
ECSS Compliance RAG Engine — Hybrid Architecture
Extends traditional vector retrieval with a Normative Document Graph.

Hybrid retrieval:
  1. Vector search (ChromaDB) → retrieves semantically similar text chunks
  2. Graph traversal (NetworkX) → retrieves structurally related cross-references
  3. Context Assembly → unified grounded context passed to the LLM
"""

import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graph_builder import ECSSGraphBuilder  

load_dotenv()

CHROMA_DB_DIR   = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL_ID    = "Qwen/Qwen2.5-7B-Instruct"

def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def merge_contexts(vector_context: str, graph_result: dict) -> str:
    """
    Combines lexical vector context with topological graph context.
    Ensures grounded retrieval by linking semantics to normative structures.
    """
    parts = ["=== LEXICAL RETRIEVAL (VECTOR SPACE) ===", vector_context]

    graph_text = graph_result.get("context_text", "")
    if graph_text:
        shall  = graph_result.get("shall_count", 0)
        should = graph_result.get("should_count", 0)
        parts.append(
            f"\n=== TOPOLOGICAL RETRIEVAL (NORMATIVE GRAPH) "
            f"(SHALL: {shall}, SHOULD: {should}) ==="
        )
        parts.append(graph_text)

    return "\n\n".join(parts)

def initialize_rag_pipeline():
    """Initialize the hybrid retrieval components."""

    # 1. Vector store 
    embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 2. Normative Document Graph
    graph_builder = ECSSGraphBuilder()

    if not graph_builder.load():
        print("[GRAPH] Cache not found — building from vectorstore (one-time)...")
        graph_builder.build_from_vectorstore(vectorstore)
        graph_builder.save()

    # 3. LLM Setup
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError("[ERROR] HUGGINGFACEHUB_API_TOKEN not found in .env")

    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_MODEL_ID,
        huggingfacehub_api_token=hf_token,
        max_new_tokens=512,
        temperature=0.1,
        repetition_penalty=1.15,
    )
    llm = ChatHuggingFace(llm=llm_endpoint, verbose=False)

    
    # 4. System Prompt Design for Explainability
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert ESA compliance assistant specialised in ECSS 
(European Cooperation for Space Standardization) standards.

The context below relies on a hybrid retrieval architecture combining lexical 
similarity (Vector Space) with relational knowledge (Normative Graph).

CRITICAL INSTRUCTIONS FOR GROUNDED GENERATION:
1. Ground your answer STRICTLY on the retrieved context. If the answer is absent, state:
   "Confidence: LOW — The retrieved ECSS documents do not contain a definitive answer."
2. Never hallucinate document numbers, section numbers, or compliance rules.
3. Explicitly cite the ECSS standard, section, and page number for every claim.
4. Distinguish clearly between mandatory constraints (SHALL), recommendations (SHOULD), and permissions (MAY).
5. EXPLAINABILITY: If your answer relies on cross-referenced requirements provided by the TOPOLOGICAL RETRIEVAL (Normative Graph) context, you MUST explicitly state at the end of your answer: 
   "Through topological graph expansion, the following interconnected requirements were also identified..." to highlight the multi-hop dependencies.

Retrieved context:
{context}"""
        ),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    def hybrid_invoke(question: str) -> str:
        """Executes the dual-branch retrieval and merges results."""
        # Branch 1: Vector Space
        vector_docs     = retriever.invoke(question)
        vector_context  = format_docs(vector_docs)

        # Branch 2: Topological Graph (Now anchored to Vector Docs!)
        graph_result    = graph_builder.query_graph(question, hops=2, vector_docs=vector_docs)

        # Merge Contexts
        full_context    = merge_contexts(vector_context, graph_result)

        # Generate Grounded Response
        answer = chain.invoke({
            "context":  full_context,
            "question": question,
        })
        return answer
    return hybrid_invoke, retriever, graph_builder