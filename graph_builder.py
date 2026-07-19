"""
ECSS Normative Document Graph Builder
──────────────────────────────────────
Builds a NetworkX normative graph from ECSS documents already loaded
into the ChromaDB vectorstore. Prevents orphaned requirements by reading
structural metadata injected during the ingestion phase.

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
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Patterns ECSS 
STANDARD_PATTERN  = re.compile(r"ECSS-[A-Z]-[A-Z]{2}-\d+[A-Z]?", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"<?\b(\d+\.\d+(?:\.\d+)*)>?\s+([A-Z][^\n]{5,60})")
SHALL_PATTERN     = re.compile(r"[^.]*\b(shall|should|may)\b[^.]*\.", re.IGNORECASE)
TERM_PATTERN      = re.compile(r"\"([^\"]{3,40})\"|\b([A-Z][A-Z\s]{3,30})\b")
GRAPH_CACHE       = Path("./ecss_graph.json")

CHROMA_DB_DIR   = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


class ECSSGraphBuilder:
    """
    Builds and queries a topological normative graph of ECSS standards.
    Uses NetworkX (in-memory), establishing structural relationships
    between sections and deontic requirements.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._node_counter = 0

    # Node helpers

    def _node_id(self, prefix: str, text: str) -> str:
        """Generates a stable node ID based on text content."""
        clean = re.sub(r"\s+", "_", text.strip().lower())[:60]
        return f"{prefix}:{clean}"

    def _add_node(self, node_id: str, label: str, node_type: str, **attrs):
        """Adds a node, or updates it if a more accurate title is found later."""
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, label=label, type=node_type, **attrs)
        else:
            # If the node was created via fallback metadata, update it when the real title arrives
            existing_title = self.graph.nodes[node_id].get("title", "")
            if attrs.get("title") and existing_title == "Inherited Section":
                self.graph.nodes[node_id]["title"] = attrs["title"]
                self.graph.nodes[node_id]["label"] = label

    def _add_edge(self, src: str, tgt: str, relation: str):
        if not self.graph.has_edge(src, tgt):
            self.graph.add_edge(src, tgt, relation=relation)

    # Extraction helpers 

    def _extract_standard(self, text: str) -> str | None:
        m = STANDARD_PATTERN.search(text)
        return m.group(0).upper() if m else None

    def _extract_sections(self, text: str) -> list[tuple[str, str]]:
        """Returns a list of (section_number, section_title)."""
        return [(m.group(1), m.group(2).strip()) for m in SECTION_PATTERN.finditer(text)]

    def _extract_requirements(self, text: str) -> list[str]:
        """Extracts phrases containing deontic modals (SHALL / SHOULD / MAY)."""
        return [m.group(0).strip() for m in SHALL_PATTERN.finditer(text)]

    def _detect_deontic(self, text: str) -> str:
        """Identifies the requirement weight: SHALL / SHOULD / MAY."""
        t = text.lower()
        if "shall" in t:
            return "SHALL"
        elif "should" in t:
            return "SHOULD"
        elif "may" in t:
            return "MAY"
        return "UNKNOWN"

    # Core builder

    def build_from_vectorstore(self, vectorstore: Chroma) -> nx.DiGraph:
        """
        Reads all chunks from the vector store and builds the topology.
        Integrates chunk metadata to prevent orphaned requirements.
        """
        print("[GRAPH] Building normative graph from vectorstore chunks...")
        
        # Retrieve all documents from the vectorstore
        collection = vectorstore._collection
        results    = collection.get(include=["documents", "metadatas"])
        docs       = results["documents"]
        metas      = results["metadatas"]

        print(f"[GRAPH] Processing {len(docs)} chunks...")

        for text, meta in zip(docs, metas):
            source   = meta.get("source", "unknown")
            page     = meta.get("page", 0)
            ecss_id  = self._extract_standard(source) or self._extract_standard(text) or "UNKNOWN"

            # Standard node
            std_id = self._node_id("std", ecss_id)
            self._add_node(std_id, ecss_id, "Standard", ecss_id=ecss_id)

           
            # ORPHAN PREVENTION: Use section metadata injected during ingestion
            
            fallback_sec = meta.get("section_id") # e.g., "§5.3.2"
            last_section_id = std_id # Default fallback is the standard itself

            if fallback_sec:
                sec_label = f"{ecss_id} {fallback_sec}"
                last_section_id = self._node_id("sec", sec_label)
                # Create a placeholder section node (will be updated if the title is found)
                self._add_node(
                    last_section_id, sec_label, "Section",
                    number=fallback_sec.replace("§", ""), title="Inherited Section",
                    ecss_id=ecss_id, page=page
                )
                self._add_edge(std_id, last_section_id, "CONTAINS")

            # Section nodes (Extract explicit headers from the text)
            sections = self._extract_sections(text)

            # Capture all sections in dense chunks
            for sec_num, sec_title in sections:   
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
            
            #Captures all requirements in the chunk
            for req_text in reqs:  
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
        print(f"[GRAPH] Normative Graph built: {n_nodes} nodes, {n_edges} edges")
        return self.graph

    def _add_cross_references(self):
        """
        Adds REFERENCES relationships between requirements that cite the same section.
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
                for nid, data in self.graph.nodes(data=True):
                    if data.get("type") == "Section" and data.get("number") == cited_sec:
                        self._add_edge(req_id, nid, "REFERENCES")

    # Query interface

    def query_graph(self, query: str, hops: int = 2, vector_docs: list = None) -> dict:
        """
        Retrieves relevant seed nodes based on the query and traverses 
        the topology. If vector_docs are provided, it strictly anchors 
        the seed nodes to the semantic vector results.
        """
        query_lower = query.lower()
        relevant_nodes = []

        # 1. FIND SEED NODES (Anchor to Vector Results)
        if vector_docs:
            for doc in vector_docs:
                source = doc.metadata.get("source", "")
                ecss_id = self._extract_standard(source) or self._extract_standard(doc.page_content) or "UNKNOWN"
                
                # Cerca le sezioni direttamente nel testo del chunk
                sections = self._extract_sections(doc.page_content)
                for sec_num, _ in sections:
                    sec_label = f"{ecss_id} §{sec_num}"
                    sec_id = self._node_id("sec", sec_label)
                    if self.graph.has_node(sec_id):
                        relevant_nodes.append(sec_id)
                
                # Method B: Fallback to the structural metadata injected during ingestion
                fallback_sec = doc.metadata.get("section_id")
                if fallback_sec:
                    # Clean the metadata if it lacks the paragraph symbol
                    if not fallback_sec.startswith("§"):
                        fallback_sec = f"§{fallback_sec}"
                    sec_label = f"{ecss_id} {fallback_sec}"
                    sec_id = self._node_id("sec", sec_label)
                    if self.graph.has_node(sec_id):
                        relevant_nodes.append(sec_id)

        # Fallbcck for tab 2 
        if not relevant_nodes:
            # Filtriamo parole molto brevi per evitare rumore
            query_words = [w for w in query_lower.split() if len(w) > 4] 
            for nid, data in self.graph.nodes(data=True):
                label = data.get("label", "").lower()
                if query_words and any(word in label for word in query_words):
                    relevant_nodes.append(nid)

        # Deduplicate initial nodes
        relevant_nodes = list(set(relevant_nodes))

        if not relevant_nodes:
            return {
                "nodes": [], "seed_nodes": [], "discovered_nodes": [],
                "context_text": "", "shall_count": 0, "should_count": 0,
            }

        # 2. EXTRACT EGO-GRAPH (Preventing Supernode Explosion)
        subgraph_nodes = set()
        
        # Creiamo una vista del grafo che IGNORA i nodi 'Standard' per evitare il fan-out estremo
        def filter_standard_nodes(n):
            return self.graph.nodes[n].get("type") != "Standard"
            
        restricted_graph = nx.subgraph_view(self.graph, filter_node=filter_standard_nodes)

        for nid in relevant_nodes[:4]:  # Matching the k=4 of vector search
            try:
                # Usiamo il restricted_graph in modo che non possa "rimbalzare" sul nodo radice
                ego = nx.ego_graph(restricted_graph, nid, radius=hops)
                subgraph_nodes.update(ego.nodes())
            except Exception:
                pass

        # 3. ASSEMBLE CONTEXT & TRACK EXPANSION TRACE
        context_parts, nodes_info, seed_info, discovered_info = [], [], [], []
        shall_count, should_count = 0, 0

        for nid in subgraph_nodes:
            data      = self.graph.nodes[nid]
            node_type = data.get("type", "")
            label     = data.get("label", "")
            
            node_dict = {"id": nid, "type": node_type, "label": label[:80]}
            nodes_info.append(node_dict)

            # Categorize for Explainability Trace
            if nid in relevant_nodes:
                seed_info.append(node_dict)
            else:
                discovered_info.append(node_dict)

            # Context formatting
            if node_type == "Requirement":
                deontic = data.get("deontic", "")
                ecss_id = data.get("ecss_id", "")
                page    = data.get("page", "")
                full    = data.get("full_text", label)

                context_parts.append(f"[{deontic}] {ecss_id} p.{page}: {full}")
                if deontic == "SHALL": shall_count += 1
                elif deontic == "SHOULD": should_count += 1
            elif node_type == "Section":
                context_parts.append(f"[SECTION] {label} (p.{data.get('page', '?')})")

        return {
            "nodes": nodes_info, "seed_nodes": seed_info, "discovered_nodes": discovered_info,
            "context_text": "\n".join(context_parts), "shall_count": shall_count, "should_count": should_count,
        }

    def get_stats(self) -> dict:
        """Graph topology statistics for operational telemetry."""
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
        """Persists the topological graph to JSON."""
        data = nx.node_link_data(self.graph)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[GRAPH] Saved to {path}")

    def load(self, path: Path = GRAPH_CACHE) -> bool:
        """Loads the graph from disk if available."""
        if not Path(path).exists():
            return False
        with open(path) as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)
        print(f"[GRAPH] Loaded from {path} — "
              f"{self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        return True