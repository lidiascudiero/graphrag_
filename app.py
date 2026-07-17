"""
Streamlit Frontend — ECSS Compliance Agent + Knowledge Graph

This application extends the base RAG architecture with a GraphRAG approach:
  - Tab 1: Hybrid Chat (Vector + Graph) with LLMOps telemetry logging.
  - Tab 2: Knowledge Graph Explorer (Network stats + interactive visual subgraph).
  - Tab 3: Graph vs Vector comparison (for portfolio demonstration).
  - Sidebar: Real-time network structural stats, hubs, and LLMOps viewer.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import time
from datetime import datetime
import tempfile

import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network

from rag_engine_graph import initialize_rag_pipeline

# Application Configuration 
st.set_page_config(
    page_title="ESA ECSS Compliance Agent + KG",
    page_icon="🛰️",
    layout="wide",
)

#Telemetry and Visualization Helper Functions 

def log_telemetry(query, answer, latency_sec, retrieved_nodes_count):
    """Saves operational logs for LLMOps analysis (latency, query, node count)."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "latency_seconds": round(latency_sec, 2),
        "graph_nodes_retrieved": retrieved_nodes_count,
        # Save only a preview of the answer for privacy compliance
        "answer_preview": answer[:100] + "..." 
    }
    
    with open("query_telemetry.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def render_interactive_subgraph(nodes, edges):
    """Creates an interactive web visualization of the subgraph using Pyvis."""
    # Initialize the network with an elegant dark theme
    net = Network(height="500px", width="100%", bgcolor="#0e1117", font_color="white")
    
    # Color mapping based on node type
    color_map = {
        "Standard": "#ff4b4b",   # Streamlit Red
        "Section": "#ffa421",    # Orange
        "Requirement": "#00ff1e",# Neon Green for requirements
        "Term": "#1f78b4"        # Blue
    }

    # 1. NODE CONFIGURATION: Short labels on the node, full text on hover
    for node in nodes:
        node_color = color_map.get(node.get("type"), "#ffffff")
        
        # Ensure we safely get the full text label
        full_text = node.get("label", str(node.get("id", "Unknown")))
        
        # Truncate text for the visible label (max 25 chars)
        short_label = full_text[:25] + "..." if len(full_text) > 25 else full_text
        
        # Make Standard and Section nodes larger than standard Requirements
        node_size = 25 if node.get("type") in ["Standard", "Section"] else 15

        net.add_node(
            node["id"], 
            label=short_label,       # The text visible directly on the node
            title=full_text,         # The tooltip text shown on mouse hover
            color=node_color,
            size=node_size
        )
        
    # 2. EDGE CONFIGURATION: Add thickness and color for visibility
    for edge in edges:
        net.add_edge(
            edge["source"], 
            edge["target"], 
            color="#555555", 
            width=2
        )

    # 3. PHYSICS CONFIGURATION: Activate repulsion to space out nodes harmoniously
    net.repulsion(node_distance=150, central_gravity=0.2, spring_length=200, spring_strength=0.05)
    net.toggle_physics(True)
    
    # Save to a temporary file and render the HTML in Streamlit
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, 'r', encoding='utf-8') as HtmlFile:
            components.html(HtmlFile.read(), height=530)

# Header 
st.title("🛰️ ECSS Compliance Assistant")
st.markdown(
    "*Hybrid RAG + Knowledge Graph system for ESA ECSS standards compliance verification.*"
)

# Load Engine 
@st.cache_resource
def load_engine():
    """Initializes and caches the heavy RAG components."""
    with st.spinner("Initializing vector store, knowledge graph and LLM..."):
        return initialize_rag_pipeline()

hybrid_invoke, retriever, graph_builder = load_engine()

# Sidebar: Graph Stats, Metrics & LLMOps 
with st.sidebar:
    st.header("📊 Knowledge Graph Stats")
    stats = graph_builder.get_stats()
    st.metric("Total Nodes", stats.get("total_nodes", 0))
    st.metric("Total Edges", stats.get("total_edges", 0))
    st.metric("SHALL Requirements", stats.get("shall_reqs", 0))
    st.metric("SHOULD Requirements", stats.get("should_reqs", 0))

    st.divider()
    st.subheader("Node Types")
    for node_type, count in stats.get("node_types", {}).items():
        st.write(f"**{node_type}**: {count}")
        
    st.divider()
    st.subheader("📈 Advanced Network Metrics")
    
    # Calculate live structural metrics using NetworkX
    if hasattr(graph_builder, 'graph') and graph_builder.graph and len(graph_builder.graph.nodes) > 0:
        G = graph_builder.graph
        density = nx.density(G)
        st.metric("Graph Density", f"{density:.4f}", help="Indicates how interconnected the normative corpus is.")
        
        # Identify the most central nodes (Normative Hubs)
        centrality = nx.degree_centrality(G)
        top_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        
        st.markdown("**Top Normative Hubs (Most referenced):**")
        for node, score in top_hubs:
            st.caption(f"🔹 {node} (Score: {score:.2f})")
    else:
        st.info("Advanced metrics unavailable. Graph might be empty.")

    #  LLMOps Telemetry Viewer 
    st.divider()
    st.subheader("🕵️ LLMOps Telemetry")
    
    # Read the log file if it exists
    log_file_path = "query_telemetry.jsonl"
    if os.path.exists(log_file_path):
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        st.metric("Total Queries Logged", len(lines))
        
        with st.expander("🔍 View Recent Logs"):
            # Show the last 3 logs in reverse chronological order
            for line in reversed(lines[-3:]):
                try:
                    log_data = json.loads(line)
                    # Format the log elegantly
                    st.markdown(f"**Query:** {log_data.get('query')}")
                    st.caption(f"⏱️ Latency: {log_data.get('latency_seconds')}s | 🕸️ Nodes: {log_data.get('graph_nodes_retrieved')}")
                    st.divider()
                except json.JSONDecodeError:
                    continue
    else:
        st.info("No telemetry logs recorded yet. Ask the agent a question to generate the first log!")


#  Tabs 
tab1, tab2, tab3 = st.tabs([
    "💬 Compliance Chat",
    "🕸️ Knowledge Graph Explorer",
    "📊 Vector vs Graph Comparison",
])


# TAB 1 Compliance Chat
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(
        "e.g., What are the software unit testing requirements according to ECSS-E-ST-40C?"
    ):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Searching ECSS corpus (vector + graph)..."):
                start_time = time.time() # Start telemetry timer

                # Vector sources (for expander)
                source_docs  = retriever.invoke(prompt)

                # Graph context (for expander) 
                graph_result = graph_builder.query_graph(prompt, hops=2)

                # Hybrid answer generation
                answer = hybrid_invoke(prompt)
                
                end_time = time.time() # End telemetry timer
                
                # Save LLMOps telemetry log
                total_reqs = graph_result.get('shall_count', 0) + graph_result.get('should_count', 0)
                log_telemetry(
                    query=prompt, 
                    answer=answer, 
                    latency_sec=(end_time - start_time), 
                    retrieved_nodes_count=total_reqs
                )

                st.markdown(answer)

                # Sources expander to show Grounding
                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("📚 Vector Search Sources"):
                        for i, doc in enumerate(source_docs):
                            source  = doc.metadata.get("source", "Unknown")
                            page    = doc.metadata.get("page", "N/A")
                            st.markdown(f"**[{i+1}]** `{source}` — p.{page}")
                            st.caption(f"_{doc.page_content[:250]}..._")

                with col2:
                    with st.expander(
                        f"🕸️ Graph Retrieved "
                        f"(SHALL: {graph_result.get('shall_count', 0)}, "
                        f"SHOULD: {graph_result.get('should_count', 0)})"
                    ):
                        for node in graph_result.get("nodes", [])[:10]:
                            icon = {
                                "Standard":    "📋",
                                "Section":     "📑",
                                "Requirement": "⚖️",
                                "Term":        "📖",
                            }.get(node.get("type"), "•")
                            st.markdown(f"{icon} **{node.get('type', 'Unknown')}** — {node.get('label', '')}")

        st.session_state.messages.append({"role": "assistant", "content": answer})



# TAB 2  Knowledge Graph Explorer

with tab2:
    st.subheader("🕸️ Explore the ECSS Knowledge Graph")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Query the graph directly**")
        kg_query = st.text_input(
            "Search term",
            placeholder="e.g. unit testing, software safety, verification",
        )
        hops = st.slider("Graph traversal depth (hops)", 1, 3, 2)

        if kg_query:
            result = graph_builder.query_graph(kg_query, hops=hops)
            nodes  = result.get("nodes", [])

            st.markdown(f"Found **{len(nodes)} nodes** connected to *'{kg_query}'*")
            st.markdown(
                f"SHALL: **{result.get('shall_count', 0)}** | "
                f"SHOULD: **{result.get('should_count', 0)}**"
            )

            # Node list filtering tool
            type_filter = st.multiselect(
                "Filter by type",
                ["Standard", "Section", "Requirement", "Term"],
                default=["Standard", "Section", "Requirement"],
            )
            for node in nodes:
                if node.get("type") in type_filter:
                    icon = {
                        "Standard":    "📋",
                        "Section":     "📑",
                        "Requirement": "⚖️",
                        "Term":        "📖",
                    }.get(node.get("type"), "•")
                    st.markdown(f"{icon} `{node.get('type')}` — {node.get('label')}")

    with col_right:
        st.markdown("**🕸️ Interactive Subgraph Visualization**")
        if kg_query:
            nodes = result.get("nodes", [])
            edges = result.get("edges", [])
            
            # Render the interactive graph if nodes are found
            if len(nodes) > 0:
                render_interactive_subgraph(nodes, edges)
                st.divider()
                
            # Render textual context below the graph
            st.markdown("**Graph context retrieved**")
            context = result.get("context_text", "")
            if context:
                # Group requirements by type (SHALL/SHOULD) and other lines
                lines = context.split("\n")
                shall_lines  = [l for l in lines if "[SHALL]"  in l]
                should_lines = [l for l in lines if "[SHOULD]" in l]
                other_lines  = [l for l in lines if "[SHALL]" not in l
                                and "[SHOULD]" not in l and l.strip()]

                if shall_lines:
                    st.markdown("**🔴 SHALL (mandatory)**")
                    for l in shall_lines[:5]:
                        st.info(l.replace("[SHALL] ", ""))

                if should_lines:
                    st.markdown("**🟡 SHOULD (recommended)**")
                    for l in should_lines[:5]:
                        st.warning(l.replace("[SHOULD] ", ""))

                if other_lines:
                    st.markdown("**📑 Sections & Standards**")
                    for l in other_lines[:5]:
                        st.markdown(f"- {l}")
            else:
                st.info("No graph context found for this query. Try different keywords.")
        else:
            st.markdown(
                "Enter a search term on the left to explore the knowledge graph."
            )



# TAB 3  Vector vs Graph Comparison 
with tab3:
    st.subheader("📊 Retrieval Comparison: Vector Search vs Knowledge Graph")
    st.markdown(
        "This tab demonstrates why Graph RAG outperforms plain vector search "
        "for compliance documents. Run the same query on both systems."
    )

    cmp_query = st.text_input(
        "Comparison query",
        placeholder="e.g. software unit testing requirements",
        key="cmp_query",
    )

    if cmp_query and st.button("🔍 Run Comparison"):
        col_v, col_g = st.columns(2)

        with col_v:
            st.markdown("### 🔵 Vector Search Only")
            with st.spinner("Running vector retrieval..."):
                v_docs = retriever.invoke(cmp_query)
                st.markdown(f"**Retrieved {len(v_docs)} chunks**")
                for i, doc in enumerate(v_docs):
                    src  = doc.metadata.get("source", "?")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i+1}** — `{src}` p.{page}")
                    st.caption(doc.page_content[:300] + "...")
                    st.divider()

        with col_g:
            st.markdown("### 🟢 Knowledge Graph Traversal")
            with st.spinner("Running graph traversal..."):
                g_result = graph_builder.query_graph(cmp_query, hops=2)
                n_nodes  = len(g_result.get("nodes", []))
                shall    = g_result.get("shall_count", 0)
                should   = g_result.get("should_count", 0)

                st.markdown(
                    f"**Found {n_nodes} connected nodes** "
                    f"(SHALL: {shall}, SHOULD: {should})"
                )

                lines = g_result.get("context_text", "").split("\n")
                for l in lines[:12]:
                    if l.strip():
                        if "[SHALL]" in l:
                            st.error(l)
                        elif "[SHOULD]" in l:
                            st.warning(l)
                        elif "[SECTION]" in l:
                            st.info(l)
                        else:
                            st.write(l)

        # Summary explanatory box for stakeholders
        st.divider()
        st.markdown("### 🎯 What the graph found that vector search missed")
        st.markdown(
            "The knowledge graph traversal recovers **cross-referenced requirements** "
            "and **structurally related sections** that are not semantically similar "
            "to the query but are normatively connected. "
            "This is critical for ECSS compliance, where a requirement in §5.3 "
            "may depend on definitions in §3.1 of a different standard."
        )
