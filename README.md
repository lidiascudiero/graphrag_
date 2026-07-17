# ECSS Compliance RAG Engine: Hybrid Graph-Vector Architecture

Transitioning from Flat Text Search to Topologically-Aware AI for Aerospace Standards

## Professional Disclaimer (NDA & Privacy)

To comply with Non-Disclosure Agreements (NDAs) and copyright restrictions regarding the official European Cooperation for Space Standardization (ECSS) corpus, the source code in this repository runs on a **Mock Dataset**. 

The proprietary industrial datasets and the full compiled databases are strictly confidential and are not included in this repository. However, the architectural logic, the knowledge graph extraction pipeline, and the metrics discussed in this documentation reflect the real-world deployment of the system tested on the complete normative corpus.

## The "Cognitive" Angle: From Flat Text to Normative Networks

> **Author's Note:** My background in Cognitive Psychology and Neuroscience allows me to treat complex regulatory frameworks not as isolated paragraphs, but as interconnected semantic networks. Just as neural pathways define cognitive functions through their connections, an aerospace standard defines compliance through its structural cross-references. By implementing a Knowledge Graph alongside a Vector Database, I apply a rigorous structural mindset to information retrieval treating every ECSS requirement as a node in a broader "normative behavior" profile.

## Project Overview

This project focuses on resolving the critical limitations of standard Vector-based RAG (Retrieval-Augmented Generation) when applied to rigid aerospace engineering standards (ECSS). 

### The Industrial Challenge:
*   **Semantic Ambiguity:** Standard vector searches fail when a requirement states *"Testing shall be performed as per §5.3"*, because the actual testing parameters are structurally linked, not semantically similar.
*   **Deontic Rigidity:** Aerospace compliance relies on the strict categorization of modal verbs (SHALL for mandatory, SHOULD for recommended, MAY for optional). Vector spaces do not inherently differentiate these constraints.
*   **Cross-Referencing:** Navigating a real-world scenario requires understanding multi-hop dependencies across different engineering domains (e.g., Software Engineering vs. Quality Assurance).

## System Architecture

The hybrid pipeline is designed to merge semantic similarity with structural awareness:

1.  **Semantic Layer (Vector Store):** High-performance chunk querying via ChromaDB and Hugging Face Embeddings (`all-mpnet-base-v2`)[cite: 2, 5].
2.  **Structural Layer (Knowledge Graph):** An in-memory NetworkX directed graph extracting hierarchies (`Standard → Section → Requirement`) and cross-references (`REFERENCES`, `DEFINES`).
3.  **Intelligence Layer:** A unified prompt merging Vector Context and Graph-Retrieved Context, parsed by the `Qwen/Qwen2.5-7B-Instruct` LLM.

## Real-World Production Metrics

While the live demo uses a mocked subset, the architecture was originally tested and validated on a full-scale ECSS corpus. Below are the actual Knowledge Graph telemetry stats from the production environment:

**Knowledge Graph Stats**
*   **Total Nodes:** 1,700
*   **Total Edges:** 1,812
*   **SHALL Requirements:** 1,087
*   **SHOULD Requirements:** 38

**Node Types Distribution**
*   **Requirement:** 1,159
*   **Section:** 514
*   **Term:** 24
*   **Standard:** 3

**Advanced Network Metrics**
*   **Graph Density:** `0.0006` *(Indicating a highly specific, sparsely connected normative topology)*
*   **Top Normative Hubs (Most referenced standards):**
    *    `std:ecss-e-st-40c` (Centrality Score: 0.40)
    *    `std:ecss-q-st-80c` (Centrality Score: 0.23)
    *    `std:ecss-i-st-30`  (Centrality Score: 0.07)
*   **LLMOps Telemetry:** Live Queries Logged during the baseline validation phase.

## Interactive ECSS Graph-RAG Demo

Experience the Hybrid Retrieval pipeline in real-time through a dedicated Streamlit application. This dashboard simulates a live compliance verification session, visualizing how both vector and graph contexts are extracted.

### Hybrid Decoding & Graph Traversal Simulation

*   **Focus:** Hybrid RAG pipeline comparing traditional ChromaDB semantic search with NetworkX topological traversal.
*   **Access the Streamlit Dashboard:** `[ Streamlit Cloud Link ]`
*   **Real-Time Simulation:** Observe the generation process as the agent processes your query across two branches (Vector + Graph) and synthesizes a compliance report distinguishing SHALL and SHOULD requirements.
*   **Interactive Subgraph Visualization:** View the local ego-graph generated for your query using the Pyvis visualizer to verify how requirements cross-reference each other.
*   **Vector vs Graph Comparison:** Inspect the dedicated comparison tab to understand what the Knowledge Graph found that the traditional Vector search missed.

## Key Achievements

*   **Dual-Layer Retrieval Engine:** Successfully merged semantic search with topological traversal to capture hidden normative dependencies.
*   **Automated Graph Extraction:** Built a deterministic parser (Regex + NetworkX) capable of extracting nodes and edges directly from vector chunks without manual annotation.
*   **Explainability & Grounding:** Integrated visual UI components (expanders) to trace every LLM assertion back to the exact ECSS standard and page number, ensuring zero-hallucination compliance].
*   **Built-in LLMOps:** Implemented a lightweight telemetry logger to track latency, retrieved graph nodes, and user queries for continuous system evaluation.

## Tech Stack

*   **Data Engine:** Python, NetworkX, ChromaDB
*   **Machine Learning / LLM:** LangChain, Hugging Face Endpoint (`Qwen2.5-7B`), Sentence-Transformers
*   **Visualization & UI:** Streamlit, Pyvis (Interactive Network Graphs)
