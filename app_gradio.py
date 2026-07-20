"""
Gradio Frontend — ECSS Compliance Agent + Normative Graph
Adapted for Hugging Face Spaces.
"""

import os
import json
import time
from datetime import datetime
import tempfile
import html

import gradio as gr
import networkx as nx
from pyvis.network import Network


from rag_engine_graph import initialize_rag_pipeline

# Global inizialization
print("Initializing vector store, normative graph and LLM...")
hybrid_invoke, retriever, graph_builder = initialize_rag_pipeline()


# 2. Support Functions & UI Logic


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

def get_telemetry_logs():
    """Reads telemetry logs to update the UI."""
    log_file_path = "query_telemetry.jsonl"
    if not os.path.exists(log_file_path):
        return "No telemetry logs recorded yet."
    
    with open(log_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    md = f"**Total Queries Logged:** {len(lines)}\n\n---\n\n"
    for line in reversed(lines[-3:]):
        try:
            log_data = json.loads(line)
            md += f"**Query:** {log_data.get('query')}\n"
            md += f"⏱️ Latency: {log_data.get('latency_seconds')}s | 🕸️ Nodes: {log_data.get('graph_nodes_retrieved')}\n\n---\n"
        except json.JSONDecodeError:
            continue
    return md

def generate_pyvis_html(nodes, edges):
    """Renders the local ego-graph using Pyvis and returns iframe HTML."""
    if not nodes:
        return "<div>No topology to display.</div>"

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
        with open(tmp_file.name, 'r', encoding='utf-8') as f:
            html_data = f.read()
    
    # Escape per l'inserimento in iframe
    escaped_html = html.escape(html_data)
    iframe = f'<iframe srcdoc="{escaped_html}" width="100%" height="530px" style="border:none;"></iframe>'
    return iframe

def get_graph_stats():
    """Generates Markdown for the static graph stats."""
    stats = graph_builder.get_stats()
    md = f"""
### 📊 Normative Graph Stats
* **Total Nodes:** {stats.get("total_nodes", 0)}
* **Total Edges:** {stats.get("total_edges", 0)}
* **SHALL Requirements:** {stats.get("shall_reqs", 0)}
* **SHOULD Requirements:** {stats.get("should_reqs", 0)}

#### Node Types
"""
    for node_type, count in stats.get("node_types", {}).items():
        md += f"* **{node_type}**: {count}\n"
    
    md += "\n#### 📈 Advanced Metrics\n"
    if hasattr(graph_builder, 'graph') and graph_builder.graph and len(graph_builder.graph.nodes) > 0:
        G = graph_builder.graph
        density = nx.density(G)
        md += f"* **Graph Density:** {density:.4f}\n\n**Top Normative Hubs:**\n"
        
        centrality = nx.degree_centrality(G)
        top_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        for node, score in top_hubs:
            md += f"🔹 {node} (Score: {score:.2f})\n"
    return md

# 3. Callbacks UI

def chat_interaction(prompt, history):
    start_time = time.time() 

    # Retrieval
    source_docs = retriever.invoke(prompt)
    graph_result = graph_builder.query_graph(prompt, hops=2, vector_docs=source_docs)
    answer = hybrid_invoke(prompt)
    
    end_time = time.time() 
    
    total_reqs = graph_result.get('shall_count', 0) + graph_result.get('should_count', 0)
    log_telemetry(prompt, answer, (end_time - start_time), total_reqs)

    # Format Lexical view
    lexical_md = "### 📚 Lexical Grounding (Vectors)\n"
    for i, doc in enumerate(source_docs):
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        lexical_md += f"**[{i+1}]** `{src}` — p.{page}\n\n_{doc.page_content[:250]}..._\n\n"

    # Format Topological view
    topo_md = f"### 🕸️ Topological Grounding (Graph)\n**SHALL:** {graph_result.get('shall_count', 0)} | **SHOULD:** {graph_result.get('should_count', 0)}\n\n"
    for node in graph_result.get("nodes", [])[:10]:
        icon = {"Standard": "📋", "Section": "📑", "Requirement": "⚖️", "Term": "📖"}.get(node.get("type"), "•")
        topo_md += f"{icon} **{node.get('type', 'Unknown')}** — {node.get('label', '')}\n\n"

    history.append((prompt, answer))
    
    return "", history, lexical_md, topo_md, get_telemetry_logs()

def explore_graph(kg_query, hops, type_filter):
    if not kg_query:
        return "<div>Enter a query</div>", "No search term provided."

    result = graph_builder.query_graph(kg_query, hops=hops)
    nodes = result.get("nodes", [])
    edges = result.get("edges", [])
    
    # Nodes filtering based on user selection
    filtered_nodes = [n for n in nodes if n.get("type") in type_filter]
    html_output = generate_pyvis_html(filtered_nodes, edges)

    # Context extraction for Markdown
    context = result.get("context_text", "")
    info_md = f"**Found {len(nodes)} nodes** | SHALL: **{result.get('shall_count', 0)}** | SHOULD: **{result.get('should_count', 0)}**\n\n---\n"
    
    if context:
        lines = context.split("\n")
        shall_lines = [l for l in lines if "[SHALL]" in l]
        should_lines = [l for l in lines if "[SHOULD]" in l]
        
        if shall_lines:
            info_md += "#### 🔴 SHALL (mandatory constraints)\n"
            for l in shall_lines[:5]:
                info_md += f"* {l.replace('[SHALL] ', '')}\n"
        if should_lines:
            info_md += "#### 🟡 SHOULD (recommendations)\n"
            for l in should_lines[:5]:
                info_md += f"* {l.replace('[SHOULD] ', '')}\n"
    else:
        info_md += "*No structural connections found.*"

    return html_output, info_md

def compare_retrieval(cmp_query):
    if not cmp_query:
        return "No query", "No query"

    # Vector
    v_docs = retriever.invoke(cmp_query)
    v_md = f"**Retrieved {len(v_docs)} semantic chunks**\n\n"
    for i, doc in enumerate(v_docs):
        src = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        v_md += f"**Hit {i+1}** — `{src}` p.{page}\n> {doc.page_content[:300]}...\n\n---\n"

    # Graph
    g_result = graph_builder.query_graph(cmp_query, hops=2, vector_docs=v_docs)
    seed_nodes = g_result.get("seed_nodes", [])
    disc_nodes = g_result.get("discovered_nodes", [])
    
    seed_labels = [f"`{n['label'][:30]}...`" for n in seed_nodes[:3]]
    disc_labels = [f"`{n['label'][:30]}...`" for n in disc_nodes[:4]]
    
    g_md = f"""### 🧠 Graph Expansion Trace
* 📍 **Semantic Seed Nodes:** {', '.join(seed_labels) if seed_labels else 'None found'}
* ⬇️ *Traversing REFERENCES & CONTAINS edges (Depth: 2)*
* 🕸️ **Discovered via Topology:** {', '.join(disc_labels) + ('...' if len(disc_nodes)>4 else '') if disc_labels else 'No extra nodes'}
* ⚖️ **Mandatory Requirements (SHALL) Recovered:** `{g_result.get('shall_count', 0)}`

---
**Raw Topological Context Extracted:**
"""
    lines = g_result.get("context_text", "").split("\n")
    for l in lines[:12]:
        if l.strip():
            g_md += f"{l}\n\n"

    return v_md, g_md

# 4. Gradio UI Layout

with gr.Blocks(title="ESA ECSS Compliance Agent + Graph", theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# 🛰️ ECSS Compliance Assistant\n*Hybrid RAG architecture enabling grounded retrieval through topological and lexical search.*")
    
    with gr.Row():
        # -SIDEBAR Equivalente 
        with gr.Column(scale=1):
            gr.Markdown(get_graph_stats())
            gr.Markdown("### 🕵️ LLMOps Telemetry")
            telemetry_box = gr.Markdown(get_telemetry_logs())
            refresh_tel_btn = gr.Button("🔄 Refresh Logs", size="sm")
            refresh_tel_btn.click(fn=get_telemetry_logs, outputs=telemetry_box)

        # MAIN CONTENT 
        with gr.Column(scale=3):
            with gr.Tabs():
                # TAB 1: Chat
                with gr.Tab("💬 Compliance Chat"):
                    chatbot = gr.Chatbot(height=400)
                    msg = gr.Textbox(placeholder="e.g., What are the software unit testing requirements according to ECSS-E-ST-40C?", label="Ask a question")
                    
                    with gr.Row():
                        with gr.Accordion("📚 Lexical Grounding (Vectors)", open=False):
                            lexical_view = gr.Markdown("Lexical context will appear here...")
                        with gr.Accordion("🕸️ Topological Grounding (Graph)", open=False):
                            topo_view = gr.Markdown("Topological context will appear here...")
                    
                    msg.submit(
                        fn=chat_interaction, 
                        inputs=[msg, chatbot], 
                        outputs=[msg, chatbot, lexical_view, topo_view, telemetry_box]
                    )

                # TAB 2: Graph Explorer
                with gr.Tab("🕸️ Normative Graph Explorer"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            kg_query = gr.Textbox(label="Search term", placeholder="e.g. unit testing, verification")
                            hops = gr.Slider(minimum=1, maximum=3, step=1, value=2, label="Graph traversal depth (hops)")
                            type_filter = gr.CheckboxGroup(
                                choices=["Standard", "Section", "Requirement", "Term"],
                                value=["Standard", "Section", "Requirement"],
                                label="Filter by node type"
                            )
                            explore_btn = gr.Button("🔍 Render Topology")
                            req_extraction = gr.Markdown()
                        
                        with gr.Column(scale=2):
                            graph_html = gr.HTML(value="<div style='text-align:center; padding:50px;'>Enter a search term and click render.</div>")
                    
                    explore_btn.click(
                        fn=explore_graph,
                        inputs=[kg_query, hops, type_filter],
                        outputs=[graph_html, req_extraction]
                    )

                # TAB 3: Comparison
                with gr.Tab("📊 Retrieval Comparison"):
                    gr.Markdown("Compare purely semantic retrieval against the hybrid topological approach.")
                    cmp_query = gr.Textbox(label="Comparison query")
                    cmp_btn = gr.Button("🔍 Run Dual Pipeline")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🔵 Lexical Retrieval (Vector-Only)")
                            vec_res = gr.Markdown("Results will appear here.")
                        with gr.Column():
                            gr.Markdown("### 🟢 Topological Traversal (Normative Graph)")
                            graph_res = gr.Markdown("Results will appear here.")
                            
                    cmp_btn.click(
                        fn=compare_retrieval,
                        inputs=[cmp_query],
                        outputs=[vec_res, graph_res]
                    )

if __name__ == "__main__":
    demo.launch()