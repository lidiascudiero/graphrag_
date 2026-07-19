| Project | Domain | Tech Stack | Highlights |
| :--- | :--- | :--- | :--- |
| **[ECSS Compliance RAG Engine](inserisci-qui-il-link)** | **Aerospace Standards (ECSS).** Overcoming semantic ambiguity and deontic rigidity (SHALL/SHOULD) in complex regulatory frameworks. | **Data Engine:** NetworkX, ChromaDB.<br>**ML/LLM:** LangChain, Hugging Face (Qwen2.5), Sentence-Transformers.<br>**Visuals:** Streamlit, Pyvis. | Developed a **Hybrid RAG Pipeline** merging semantic vector search with a **Normative Document Graph**. Built an interactive **Sub-graph Explorer** ensuring **grounded retrieval** and resolving topological normative cross-references. |

***

# ECSS Compliance RAG Engine: Hybrid Graph-Vector Architecture
*Transitioning from Flat Text Search to Topologically-Aware AI for Aerospace Standards*

## Professional Disclaimer (NDA)
*To comply with Non-Disclosure Agreements (NDAs) and copyright restrictions regarding the official European Cooperation for Space Standardization (ECSS) corpus, the source code in this repository runs on a **Mock Dataset**.*

*The proprietary industrial datasets and the full compiled databases are strictly confidential and are not included in this repository. However, the architectural logic, the knowledge graph extraction pipeline, and the metrics discussed in this documentation reflect the real-world deployment of the system tested on the complete normative corpus.*

---

## The Cognitive Angle: From Flat Text to Normative Networks

> **Author's Note:** My background in Cognitive Psychology and Neuroscience influenced the design of the retrieval architecture by emphasizing relational rather than purely lexical representations of knowledge. This allowed me to treat complex regulatory frameworks not as isolated paragraphs, but as interconnected normative networks. Just as neural pathways define cognitive functions through their connections, an aerospace standard defines compliance through its structural cross-references. By implementing a Normative Document Graph alongside a Vector Database, I apply a rigorous structural mindset to information retrieval, treating every ECSS requirement as a node in a broader "normative behavior" profile.

## Project Overview

This project focuses on resolving the critical limitations of standard Vector-based RAG (Retrieval-Augmented Generation) when applied to rigid aerospace engineering standards (ECSS).

### The Industrial Challenge:
*   **Semantic Ambiguity:** Standard vector searches fail when a requirement states *"Testing shall be performed as per §5.3"*, because the actual testing parameters are structurally linked, not semantically similar.
*   **Deontic Rigidity:** Aerospace compliance relies on the strict categorization of modal verbs (SHALL for mandatory, SHOULD for recommended, MAY for optional). Vector spaces do not inherently differentiate these constraints.
*   **Cross-Referencing:** Navigating a real-world scenario requires understanding multi-hop dependencies across different engineering domains (e.g., Software Engineering vs. Quality Assurance).

## System Architecture

The hybrid pipeline is designed to merge semantic similarity with structural awareness.

### The Hybrid Retrieval Algorithm
1.  **Top-K Semantic Search (Vector Space):** Querying ChromaDB using Hugging Face Embeddings (`all-mpnet-base-v2`) to retrieve lexically similar chunks.
2.  **Node Anchoring (Seed Nodes):** Using the retrieved chunk metadata (Standard/Section) as entry points into the NetworkX graph.
3.  **Topological Traversal (Graph Expansion):** Navigating the local ego-graph strictly along `CONTAINS` and `REFERENCES` edges, intentionally filtering out standard-level supernodes to prevent "fan-out" explosions.
4.  **Context Assembly:** Merging the lexical text with the discovered multi-hop requirements.
5.  **Grounded Generation:** Passing the enriched context to the `Qwen/Qwen2.5-7B-Instruct` LLM with explicit instructions to trace assertions back to their topological or lexical origin.

## Real-World Production Metrics

While the live demo uses a mocked subset, the architecture was originally tested and validated on a full-scale ECSS corpus. Below are the actual topology telemetry stats from the production environment:

### Normative Graph Stats
*   **Total Nodes:** 1935
*   **Total Edges:** 2098
*   **SHALL Requirements:** 1088
*   **SHOULD Requirements:** 39

### Node Types Distribution
*   **Requirement:** 1162
*   **Section:** 743
*   **Term:** 27
*   **Standard:** 3

### Advanced Network Metrics
*   **Graph Density:** `0.0006` *(Indicating a highly specific, sparsely connected normative topology)*
*   **Top Normative Hubs (Most referenced standards):**
    *   `std:ecss-q-st-80c` (Centrality Score: 0.18)
    *   `std:ecss-e-st-40c` (Centrality Score: 0.16)
    *   `std:ecsse-st-40c_§2.1` (Centrality Score: 0.06)

## Interactive ECSS Graph-RAG Demo

Experience the Hybrid Retrieval pipeline in real-time through a dedicated Streamlit application. This dashboard simulates a live compliance verification session, visualizing how both vector and graph contexts are extracted.

### Hybrid Decoding & Graph Traversal Simulation
*   **Access the Streamlit Dashboard:** `[ Streamlit Cloud Link ]`
*   **Real-Time Simulation:** Observe the generation process as the agent processes your query across two branches (Vector + Graph) and synthesizes a compliance report distinguishing SHALL and SHOULD requirements.
*   **Interactive Subgraph Visualization:** View the local ego-graph generated for your query using the Pyvis visualizer to verify how requirements cross-reference each other.
*   **Explainability Trace (Vector vs Graph):** Inspect the dedicated comparison tab to see exactly which "Seed Nodes" were triggered by the vector search, and which hidden normative dependencies were successfully "Discovered" via topological expansion.

## Key Achievements
*   **Dual-Layer Retrieval Engine:** Successfully merged lexical search with topological traversal to capture hidden normative cross-references entirely missed by flat vector spaces.
*   **Structural Metadata Injection:** Engineered an ingestion pipeline that injects section IDs directly into vector chunks, completely preventing the "orphaned requirements" issue during graph building.
*   **Automated Graph Extraction:** Built a deterministic parser (Regex + NetworkX) capable of extracting nodes and edges directly from vector chunks without manual annotation.
*   **Explainability & Grounded Retrieval:** Replaced absolute "zero hallucination" concepts with strict Grounded Retrieval mechanisms. Integrated UI components to trace every LLM assertion back to the exact ECSS standard, distinguishing between semantic hits and topological discoveries.
*   **Built-in LLMOps:** Implemented a lightweight telemetry logger to track latency, retrieved graph nodes, and user queries for continuous system evaluation.

## Tech Stack
*   **Data Engine:** Python, NetworkX, ChromaDB
*   **Machine Learning / LLM:** LangChain, Hugging Face Endpoint (`Qwen2.5-7B`), Sentence-Transformers
*   **Visualization & UI:** Streamlit, Pyvis (Interactive Network Graphs)
