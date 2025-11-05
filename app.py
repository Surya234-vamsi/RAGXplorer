# ---app.py---

import os
import streamlit as st

# Set a default USER_AGENT for web requests (prevents loader warnings)
os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")

st.set_page_config(page_title="MULTI-RAG APP", page_icon="📚", layout='wide')
st.title("Multi-Version RAG Suite 📚")


st.markdown("""
Welcome to the **Multi-Version RAG Suite**! This application allows you to interact with multiple versions of a Retrieval-Augmented Generation (RAG) system.
Use the sidebar pages to explore:
            
            - **NAIVE RAG** : Experience the standard RAG setup.
            - **CONVERSATIONAL RAG** : Engage in a dialogue with the RAG system.
            - **HYBRID / MODULAR RAG** : Helps for longer documents and enhanced context handling.
            - **ADVANCED HYBRID RAG** : Explore advanced features and configurations with LLM Knowledge.
            - **AGENTIC RAG** : Interact with an agent-based RAG system for dynamic responses.
            - **RAG with Spacy Knowledge Graph** : Leverage Spacy's capabilities for knowledge graph integration.

Each version lets you connect your own Google Generative AI API key for personalized experiences/ Custom Data Sources.            
""")


st.sidebar.text("© 2025 RAG-Suite by Surya Vamsi")

# streamlit run app.py