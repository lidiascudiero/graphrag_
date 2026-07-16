"""
Streamlit Frontend — ECSS Compliance Agent (FIXED)
────────────────────────────────────────────────────
Fix: import updated from rag_engine_fixed instead of rag_engine.
Everything else unchanged.
"""

import streamlit as st
from rag_engine_fixed import initialize_rag_pipeline   # ← only change

st.set_page_config(
    page_title="ESA ECSS Compliance Agent",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ ECSS Compliance Assistant (RAG)")
st.markdown(
    "*AI-powered validation against European Cooperation for Space Standardization (ECSS) guidelines.*"
)

@st.cache_resource
def load_engine():
    with st.spinner("Initializing ECSS Vector Space and LLM engine..."):
        return initialize_rag_pipeline()

rag_chain, retriever = load_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(
    "e.g., What are the software unit testing requirements according to ECSS-E-ST-40C?"
):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching ECSS corpus and evaluating compliance boundaries..."):
            source_docs = retriever.invoke(prompt)
            answer      = rag_chain.invoke(prompt)

            st.markdown(answer)

            with st.expander("📚 View Retrieved ECSS Grounding Sources"):
                for i, doc in enumerate(source_docs):
                    source  = doc.metadata.get("source", "Unknown ECSS Standard")
                    page    = doc.metadata.get("page", "N/A")
                    section = doc.metadata.get("section", "")
                    label   = f"**Source {i+1}:** `{source}` — p.{page}"
                    if section:
                        label += f" | {section}"
                    st.markdown(label)
                    st.caption(f"_{doc.page_content[:300]}..._")

    st.session_state.messages.append({"role": "assistant", "content": answer})
