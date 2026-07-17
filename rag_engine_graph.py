"""
ECSS Compliance RAG Engine — with Knowledge Graph (Hybrid)
Extends rag_engine_fixed.py with a second retrieval layer
based on a NetworkX Knowledge Graph.

Hybrid retrieval:
  1. Vector search (ChromaDB) → semantically similar chunks
  2. Graph traversal (NetworkX) → structurally related requirements
  3. Merge → unified context passed to the LLM

"""

import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
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
    """It combines the vector context with the graph context.
    The graph adds structurally related SHALL/SHOULD requirements
    that vector search might have missed.
    """
    parts = ["=== VECTOR SEARCH CONTEXT ===", vector_context]

    graph_text = graph_result.get("context_text", "")
    if graph_text:
        shall  = graph_result.get("shall_count", 0)
        should = graph_result.get("should_count", 0)
        parts.append(
            f"\n=== GRAPH-RETRIEVED REQUIREMENTS "
            f"(SHALL: {shall}, SHOULD: {should}) ==="
        )
        parts.append(graph_text)

    return "\n\n".join(parts)


def initialize_rag_pipeline():
    """Initialize:
      - ChromaDB vectorstore (semantic retrieval)
      - ECSSGraphBuilder (structural retrieval)
      - ChatHuggingFace LLM
      - Prompt with instructions on SHALL/SHOULD/MAY
    
    Returns: (hybrid_invoke_fn, retriever, graph_builder)
    """

    # 1. Vector store 
    embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 2. Knowledge Graph 
    graph_builder = ECSSGraphBuilder()

    # Try loading from cache if it doesn't exist, build from the vectorstore.
    if not graph_builder.load():
        print("[GRAPH] Cache not found — building from vectorstore (one-time)...")
        graph_builder.build_from_vectorstore(vectorstore)
        graph_builder.save()

    # 3. LLM
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

    #  4. Prompt 
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert ESA compliance assistant specialised in ECSS 
(European Cooperation for Space Standardization) standards.

The context below contains TWO types of retrieved information:
1. VECTOR SEARCH CONTEXT — semantically similar text chunks
2. GRAPH-RETRIEVED REQUIREMENTS — structurally related SHALL/SHOULD/MAY 
   requirements found by traversing the ECSS knowledge graph

Use BOTH sources to give a comprehensive compliance answer.

CRITICAL INSTRUCTIONS:
1. If the answer is NOT in the context, say:
   "Confidence: LOW — The retrieved ECSS documents do not contain a definitive answer."
   Never invent section numbers.
2. Always cite the specific ECSS standard and page number.
3. Explicitly distinguish SHALL (mandatory), SHOULD (recommended), MAY (optional).
4. If the graph found cross-referenced requirements, mention them.
5. Keep responses technical, precise, and actionable.

Retrieved context (vector + graph):
{context}"""
        ),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    #  5. Hybrid invoke function 
    def hybrid_invoke(question: str) -> str:
        """
        Hybrid pipeline:
          1. Retrieve vector chunks
          2. Traverse the graph
          3. Merge contexts
          4. Generate response
        """
        # Branch 1: vector
        vector_docs     = retriever.invoke(question)
        vector_context  = format_docs(vector_docs)

        # Branch 2: graph
        graph_result    = graph_builder.query_graph(question, hops=2)

        # Merge
        full_context    = merge_contexts(vector_context, graph_result)

        # Generate
        answer = chain.invoke({
            "context":  full_context,
            "question": question,
        })
        return answer

    return hybrid_invoke, retriever, graph_builder
