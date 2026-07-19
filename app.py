"""
Streamlit Frontend — ECSS Compliance Agent + Normative Graph

This application extends the base RAG architecture with a topological retrieval approach:
  - Tab 1: Hybrid Chat (Vector + Graph) with LLMOps telemetry logging.
  - Tab 2: Normative Graph Explorer (Network stats + interactive visual subgraph).
  - Tab 3: Graph vs Vector comparison.
  - Sidebar: Real-time network structural stats, hubs, and LLMOps viewer.
"""

import os
import json
import time
from datetime import datetime
import tempfile

import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network

from rag_engine_graph import initialize_rag_pipeline

st.set_page_config(
    page_title="ESA ECSS Compliance Agent + Graph",
    page_icon="🛰️",
    layout="wide",
)

def log_telemetry(query, answer, latency_sec, retrieved_nodes_count):
    """Saves operational logs for LLMOps evaluation."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "latency_seconds": round(latency_sec, 2),
        "graph_nodes_retrieved": retrieved_nodes_count,
        "answer_preview": answer[:100] + "..." 
    }
    
    with open("query_telemetry.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def render_interactive_subgraph(nodes, edges):
    """Renders the local ego-graph using Pyvis."""
    net = Network(height="500px", width="100%", bgcolor="#0e1117", font_color="white")
    
    color_map = {
        "Standard": "#ff4b4b",   
        "Section": "#ffa421",    
        "Requirement": "#00ff1e",
        "Term": "#1f78b4"        
    }

    for node in nodes:
        node_color = color_map.get(node.get("type"), "#ffffff")
        full_text = node.get("label", str(node.get("id", "Unknown")))
        short_label = full_text[:25] + "..." if len(full_text) > 25 else full_text
        node_size = 25 if node.get("type") in ["Standard", "Section"] else 15

        net.add_node(
            node["id"], 
            label=short_label,       
            title=full_text,         
            color=node_color,
            size=node_size
        )
        
    for edge in edges:
        net.add_edge(edge["source"], edge["target"], color="#555555", width=2)

    net.repulsion(node_distance=150, central_gravity=0.2, spring_length=200, spring_strength=0.05)
    net.toggle_physics(True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, 'r', encoding='utf-8') as HtmlFile:
            components.html(HtmlFile.read(), height=530)

st.title("🛰️ ECSS Compliance Assistant")
st.markdown(
    "*Hybrid RAG architecture enabling grounded retrieval through topological and lexical search.*"
)

@st.cache_resource
def load_engine():
    with st.spinner("Initializing vector store, normative graph and LLM..."):
        return initialize_rag_pipeline()

hybrid_invoke, retriever, graph_builder = load_engine()

with st.sidebar:
    st.header("📊 Normative Graph Stats")
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
    
    if hasattr(graph_builder, 'graph') and graph_builder.graph and len(graph_builder.graph.nodes) > 0:
        G = graph_builder.graph
        density = nx.density(G)
        st.metric("Graph Density", f"{density:.4f}", help="Topology density of the normative corpus.")
        
        centrality = nx.degree_centrality(G)
        top_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        
        st.markdown("**Top Normative Hubs (Most referenced):**")
        for node, score in top_hubs:
            st.caption(f"🔹 {node} (Score: {score:.2f})")
    else:
        st.info("Advanced metrics unavailable.")

    st.divider()
    st.subheader("🕵️ LLMOps Telemetry")
    
    log_file_path = "query_telemetry.jsonl"
    if os.path.exists(log_file_path):
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        st.metric("Total Queries Logged", len(lines))
        
        with st.expander("🔍 View Recent Logs"):
            for line in reversed(lines[-3:]):
                try:
                    log_data = json.loads(line)
                    st.markdown(f"**Query:** {log_data.get('query')}")
                    st.caption(f"⏱️ Latency: {log_data.get('latency_seconds')}s | 🕸️ Nodes: {log_data.get('graph_nodes_retrieved')}")
                    st.divider()
                except json.JSONDecodeError:
                    continue
    else:
        st.info("No telemetry logs recorded yet.")

tab1, tab2, tab3 = st.tabs([
    "💬 Compliance Chat",
    "🕸️ Normative Graph Explorer",
    "📊 Retrieval Comparison",
])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("e.g., What are the software unit testing requirements according to ECSS-E-ST-40C?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Traversing ECSS corpus (vector + graph)..."):
                start_time = time.time() 

                source_docs  = retriever.invoke(prompt)
                graph_result = graph_builder.query_graph(prompt, hops=2, vector_docs=source_docs)
                answer = hybrid_invoke(prompt)
                
                end_time = time.time() 
                
                total_reqs = graph_result.get('shall_count', 0) + graph_result.get('should_count', 0)
                log_telemetry(prompt, answer, (end_time - start_time), total_reqs)

                st.markdown(answer)

                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("📚 Lexical Grounding (Vectors)"):
                        for i, doc in enumerate(source_docs):
                            source  = doc.metadata.get("source", "Unknown")
                            page    = doc.metadata.get("page", "N/A")
                            st.markdown(f"**[{i+1}]** `{source}` — p.{page}")
                            st.caption(f"_{doc.page_content[:250]}..._")

                with col2:
                    with st.expander(
                        f"🕸️ Topological Grounding (Graph) "
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

with tab2:
    st.subheader("🕸️ Explore the Normative Graph")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Query the topology directly**")
        kg_query = st.text_input("Search term", placeholder="e.g. unit testing, verification")
        hops = st.slider("Graph traversal depth (hops)", 1, 3, 2)

        if kg_query:
            result = graph_builder.query_graph(kg_query, hops=hops)
            nodes  = result.get("nodes", [])

            st.markdown(f"Found **{len(nodes)} nodes** in the local ego-graph")
            st.markdown(f"SHALL: **{result.get('shall_count', 0)}** | SHOULD: **{result.get('should_count', 0)}**")

            type_filter = st.multiselect(
                "Filter by node type",
                ["Standard", "Section", "Requirement", "Term"],
                default=["Standard", "Section", "Requirement"],
            )
            for node in nodes:
                if node.get("type") in type_filter:
                    st.markdown(f"• `{node.get('type')}` — {node.get('label')}")

    with col_right:
        st.markdown("**🕸️ Subgraph Visualization**")
        if kg_query:
            nodes = result.get("nodes", [])
            edges = result.get("edges", [])
            
            if len(nodes) > 0:
                render_interactive_subgraph(nodes, edges)
                st.divider()
                
            st.markdown("**Structural context retrieved**")
            context = result.get("context_text", "")
            if context:
                lines = context.split("\n")
                shall_lines  = [l for l in lines if "[SHALL]"  in l]
                should_lines = [l for l in lines if "[SHOULD]" in l]
                other_lines  = [l for l in lines if "[SHALL]" not in l and "[SHOULD]" not in l and l.strip()]

                if shall_lines:
                    st.markdown("**🔴 SHALL (mandatory constraints)**")
                    for l in shall_lines[:5]:
                        st.info(l.replace("[SHALL] ", ""))

                if should_lines:
                    st.markdown("**🟡 SHOULD (recommendations)**")
                    for l in should_lines[:5]:
                        st.warning(l.replace("[SHOULD] ", ""))
            else:
                st.info("No structural connections found.")
        else:
            st.markdown("Enter a search term to render the topology.")

# TAB 3  Vector vs Graph Comparison 
with tab3:
    st.subheader("📊 Retrieval Architecture Comparison")
    st.markdown(
        "Compare purely semantic retrieval against the hybrid topological approach. "
        "Notice how the graph expands beyond exact semantic matches to recover hidden normative dependencies."
    )

    cmp_query = st.text_input("Comparison query", key="cmp_query")

    if cmp_query and st.button("🔍 Run Dual Pipeline"):
        col_v, col_g = st.columns(2)

        with col_v:
            st.markdown("### 🔵 Lexical Retrieval (Vector-Only)")
            with st.spinner("Querying vector space..."):
                v_docs = retriever.invoke(cmp_query)
                st.markdown(f"**Retrieved {len(v_docs)} semantic chunks**")
                for i, doc in enumerate(v_docs):
                    src  = doc.metadata.get("source", "?")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Hit {i+1}** — `{src}` p.{page}")
                    st.caption(doc.page_content[:300] + "...")
                    st.divider()

        with col_g:
            st.markdown("### 🟢 Topological Traversal (Normative Graph)")
            with st.spinner("Traversing normative network..."):
                g_result = graph_builder.query_graph(cmp_query, hops=2, vector_docs=v_docs)
                
                # --- NEW: GRAPH EXPANSION TRACE UI ---
                seed_nodes = g_result.get("seed_nodes", [])
                disc_nodes = g_result.get("discovered_nodes", [])
                
                with st.container(border=True):
                    st.markdown("**🧠 Graph Expansion Trace**")
                    
                    seed_labels = [f"`{n['label'][:30]}...`" for n in seed_nodes[:3]]
                    st.markdown(f"📍 **Semantic Seed Nodes:** {', '.join(seed_labels) if seed_labels else 'None found'}")
                    
                    st.markdown("⬇️ *Traversing REFERENCES & CONTAINS edges (Depth: 2)*")
                    
                    disc_labels = [f"`{n['label'][:30]}...`" for n in disc_nodes[:4]]
                    disc_str = ', '.join(disc_labels) + ("..." if len(disc_nodes)>4 else "")
                    st.markdown(f"🕸️ **Discovered via Topology:** {disc_str if disc_labels else 'No extra nodes'}")
                    
                    st.markdown(f"⚖️ **Mandatory Requirements (SHALL) Recovered:** `{g_result.get('shall_count', 0)}`")
                # -------------------------------------

                st.divider()
                st.markdown("**Raw Topological Context Extracted:**")
                lines = g_result.get("context_text", "").split("\n")
                for l in lines[:12]:
                    if l.strip():
                        if "[SHALL]" in l: st.error(l)
                        elif "[SHOULD]" in l: st.warning(l)
                        elif "[SECTION]" in l: st.info(l)
                        else: st.write(l)

      