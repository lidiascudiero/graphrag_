"""
ECSS Knowledge Graph Builder
──────────────────────────────
Builds a NetworkX knowledge graph from ECSS documents already loaded
into the ChromaDB vectorstore. No external setup required — runs in-memory.

Nodes:
  Standard    → e.g., ECSS-E-ST-40C
  Section     → e.g., "5.3.2 Unit Testing"
  Requirement → text containing SHALL / SHOULD / MAY
  Term        → terms defined in "definition" sections

Relationships:
  CONTAINS    Standard→Section, Section→Requirement
  REFERENCES  Requirement→Requirement (cross-section)
  DEFINES     Section→Term
  REQUIRES    Requirement→Activity

Usage:
  from graph_builder import ECSSGraphBuilder
  builder = ECSSGraphBuilder()
  builder.build_from_vectorstore(vectorstore)
  G = builder.graph
  results = builder.query_graph("unit testing")
"""

import re
import json
import os
from pathlib import Path
import networkx as nx
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

#Patterns ECSS 
STANDARD_PATTERN  = re.compile(r"ECSS-[A-Z]-[A-Z]{2}-\d+[A-Z]?", re.IGNORECASE)
SECTION_PATTERN   = re.compile(r"\b(\d+\.\d+(?:\.\d+)*)\s+([A-Z][^\n]{5,60})")
SHALL_PATTERN     = re.compile(r"[^.]*\b(shall|should|may)\b[^.]*\.", re.IGNORECASE)
TERM_PATTERN      = re.compile(r"\"([^\"]{3,40})\"|\b([A-Z][A-Z\s]{3,30})\b")
GRAPH_CACHE       = Path("./ecss_graph.json")

CHROMA_DB_DIR   = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


class ECSSGraphBuilder:
    """
    Builds and queries a knowledge graph of ECSS standards.
    Uses NetworkX (in-memory) no server required.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._node_counter = 0

    # Node helpers

    def _node_id(self, prefix: str, text: str) -> str:
        """Genera un ID nodo stabile dal testo."""
        clean = re.sub(r"\s+", "_", text.strip().lower())[:60]
        return f"{prefix}:{clean}"

    def _add_node(self, node_id: str, label: str, node_type: str, **attrs):
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, label=label, type=node_type, **attrs)

    def _add_edge(self, src: str, tgt: str, relation: str):
        if not self.graph.has_edge(src, tgt):
            self.graph.add_edge(src, tgt, relation=relation)

    # Extraction helpers 

    def _extract_standard(self, text: str) -> str | None:
        m = STANDARD_PATTERN.search(text)
        return m.group(0).upper() if m else None

    def _extract_sections(self, text: str) -> list[tuple[str, str]]:
        """Restituisce lista di (numero_sezione, titolo)."""
        return [(m.group(1), m.group(2).strip()) for m in SECTION_PATTERN.finditer(text)]

    def _extract_requirements(self, text: str) -> list[str]:
        """Estrae frasi con SHALL / SHOULD / MAY."""
        return [m.group(0).strip() for m in SHALL_PATTERN.finditer(text)]

    def _detect_deontic(self, text: str) -> str:
        """Identifica il tipo di requisito: SHALL / SHOULD / MAY."""
        t = text.lower()
        if "shall" in t:
            return "SHALL"
        elif "should" in t:
            return "SHOULD"
        elif "may" in t:
            return "MAY"
        return "UNKNOWN"

    #  Core builder

    def build_from_vectorstore(self, vectorstore: Chroma) -> nx.DiGraph:
        """
        Reads all chunks from the vector store and builds the graph.
        Called only once, the graph is then in memory.
        """
        print("[GRAPH] Building knowledge graph from vectorstore chunks...")
        
        # Recupera tutti i documenti dal vectorstore
        collection = vectorstore._collection
        results    = collection.get(include=["documents", "metadatas"])
        docs       = results["documents"]
        metas      = results["metadatas"]

        print(f"[GRAPH] Processing {len(docs)} chunks...")

        for text, meta in zip(docs, metas):
            source   = meta.get("source", "unknown")
            page     = meta.get("page", 0)
            ecss_id  = self._extract_standard(source) or self._extract_standard(text) or "UNKNOWN"

            #  Standard node
            std_id = self._node_id("std", ecss_id)
            self._add_node(std_id, ecss_id, "Standard", ecss_id=ecss_id)

            #  Section nodes
            sections = self._extract_sections(text)
            last_section_id = std_id

            for sec_num, sec_title in sections[:3]:   # max 3 for chunk
                sec_label = f"{ecss_id} §{sec_num}"
                sec_id    = self._node_id("sec", sec_label)
                self._add_node(
                    sec_id, sec_label, "Section",
                    number=sec_num, title=sec_title,
                    ecss_id=ecss_id, page=page
                )
                self._add_edge(std_id, sec_id, "CONTAINS")
                last_section_id = sec_id

                # Term nodes (from definition sections)
                if "definition" in sec_title.lower() or "term" in sec_title.lower():
                    for tm in TERM_PATTERN.finditer(text):
                        term = (tm.group(1) or tm.group(2) or "").strip()
                        if len(term) > 3:
                            term_id = self._node_id("term", term)
                            self._add_node(term_id, term, "Term")
                            self._add_edge(sec_id, term_id, "DEFINES")

            # Requirement nodes
            reqs = self._extract_requirements(text)
            for req_text in reqs[:5]:  # max 5 for chunk
                deontic = self._detect_deontic(req_text)
                req_id  = self._node_id("req", req_text[:80])
                self._add_node(
                    req_id, req_text[:120], "Requirement",
                    deontic=deontic, ecss_id=ecss_id,
                    page=page, full_text=req_text
                )
                self._add_edge(last_section_id, req_id, "CONTAINS")

        # Cross-references (REFERENCES) 
        self._add_cross_references()

        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print(f"[GRAPH] Graph built: {n_nodes} nodes, {n_edges} edges")
        return self.graph

    def _add_cross_references(self):
        """
        It adds REFERENCES relationships between requirements that cite the same section or the same term.
        """
        req_nodes = [
            (nid, data) for nid, data in self.graph.nodes(data=True)
            if data.get("type") == "Requirement"
        ]
        sec_pattern = re.compile(r"\d+\.\d+(?:\.\d+)*")

        for req_id, req_data in req_nodes:
            full_text = req_data.get("full_text", "")
            cited_sections = sec_pattern.findall(full_text)

            for cited_sec in cited_sections:
                # Search for Section nodes with that number
                for nid, data in self.graph.nodes(data=True):
                    if data.get("type") == "Section" and data.get("number") == cited_sec:
                        #Add REFERENCES from the requirement to the cited section.
                        self._add_edge(req_id, nid, "REFERENCES")

    # Query interface

    def query_graph(self, query: str, hops: int = 2) -> dict:
        """
        Given a query text, find the relevant nodes and traverse the graph to retrieve structural context.
        
        Returns:
            dict con 'nodes', 'context_text', 'shall_count', 'should_count'
        """
        query_lower = query.lower()
        relevant_nodes = []

        #1.Find nodes with a label similar to the query.
        for nid, data in self.graph.nodes(data=True):
            label = data.get("label", "").lower()
            if any(word in label for word in query_lower.split() if len(word) > 3):
                relevant_nodes.append(nid)

        if not relevant_nodes:
            return {
                "nodes": [],
                "context_text": "",
                "shall_count": 0,
                "should_count": 0,
            }

        # 2.ego graph (k-hop subgraph)
        subgraph_nodes = set()
        for nid in relevant_nodes[:5]:  # max 5 seed nodes
            try:
                ego = nx.ego_graph(self.graph, nid, radius=hops)
                subgraph_nodes.update(ego.nodes())
            except Exception:
                pass

        # 3. Construct textual context
        context_parts  = []
        shall_count    = 0
        should_count   = 0
        nodes_info     = []

        for nid in subgraph_nodes:
            data      = self.graph.nodes[nid]
            node_type = data.get("type", "")
            label     = data.get("label", "")

            if node_type == "Requirement":
                deontic = data.get("deontic", "")
                ecss_id = data.get("ecss_id", "")
                page    = data.get("page", "")
                full    = data.get("full_text", label)

                context_parts.append(
                    f"[{deontic}] {ecss_id} p.{page}: {full}"
                )
                if deontic == "SHALL":
                    shall_count += 1
                elif deontic == "SHOULD":
                    should_count += 1

            elif node_type == "Section":
                context_parts.append(
                    f"[SECTION] {label} (p.{data.get('page', '?')})"
                )

            nodes_info.append({
                "id":    nid,
                "type":  node_type,
                "label": label[:80],
            })

        return {
            "nodes":        nodes_info,
            "context_text": "\n".join(context_parts),
            "shall_count":  shall_count,
            "should_count": should_count,
        }

    def get_stats(self) -> dict:
        """Graph statistics for visualization in the portfolio."""
        type_counts = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        shall  = sum(1 for _, d in self.graph.nodes(data=True)
                     if d.get("deontic") == "SHALL")
        should = sum(1 for _, d in self.graph.nodes(data=True)
                     if d.get("deontic") == "SHOULD")

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types":  type_counts,
            "shall_reqs":  shall,
            "should_reqs": should,
        }

    # Persistence

    def save(self, path: Path = GRAPH_CACHE):
        """Save the graph to disk as JSON (avoids rebuild on every startup)."""
        data = nx.node_link_data(self.graph)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[GRAPH] Saved to {path}")

    def load(self, path: Path = GRAPH_CACHE) -> bool:
        """Load the graph from disk. Returns True if the file exists."""
        if not Path(path).exists():
            return False
        with open(path) as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)
        print(f"[GRAPH] Loaded from {path} — "
              f"{self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        return True
