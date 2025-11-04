# --- utils/core_utils.py ---


import os
import spacy
import google.auth
import networkx as nx
import streamlit as st
from collections import deque
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.graphs import NetworkxEntityGraph


# --- JOIN CONTEXT FUNCTION ---
def join_splits(docs):
    return "\n".join(doc.page_content for doc in docs)

# --- KG Search ---
def kg_search(q, nlp, G):
    q_tokens = [t.text.lower() for t in nlp(q)]
    hits = [n for n in G.nodes if any(tok in n.lower() for tok in q_tokens)]
    if hits:
        rels = [(a, b) for a, b in G.edges() if a in hits or b in hits]
        return f"Entities: {hits[:3]} | Relations: {rels[:5]}"
    return "No related entities found."

