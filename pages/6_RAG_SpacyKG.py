# ---pages/6_RAG_SpacyKG.py---

# !python -m spacy download en_core_web_sm

import streamlit as st
import langchain
import os, spacy, networkx as nx
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent
from langchain_community.graphs import NetworkxEntityGraph
import google.auth
from utils.core_utils import kg_search

# --- Page Config ---
st.set_page_config(page_title="Agentic RAG + KG", layout="wide")

# --- Theme Toggle ---
if "theme" not in st.session_state:
    st.session_state.theme = "light"

theme_toggle = st.sidebar.toggle("🌗 Toggle Dark Mode", value=st.session_state.theme == "dark")
st.session_state.theme = "dark" if theme_toggle else "light"

if st.session_state.theme == "dark":
    st.markdown(
        """
        <style>
        body, .stApp { background-color: #0e1117; color: #fafafa; }
        .stTextInput, .stTextArea, .stButton button {
            background-color: #1e2229; color: #fafafa; border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        body, .stApp { background-color: #ffffff; color: #000000; }
        .stTextInput, .stTextArea, .stButton button {
            background-color: #f5f5f5; color: #000000; border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Title ---
st.title("🧠 Agentic RAG + spaCy Knowledge Graph")
st.caption("Chat with your intelligent RAG agent that uses spaCy NER + KG reasoning.")

# --- API Key Setup ---
api_key = st.text_input("🔑 Enter your Gemini API key", type="password")

# # --- LOAD ENV VARIABLES ---
# load_dotenv()  # Loads variables from .env file
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- Sidebar: Document Loader ---
st.sidebar.header("📎 Load Documents")
num_urls = st.sidebar.number_input("How many URLs do you want to add?", 1, 10, 2)
urls = []
for i in range(num_urls):
    url = st.sidebar.text_input(f"Enter URL #{i+1}", "")
    if url:
        urls.append(url.strip())

build = st.sidebar.button("🔧 Build RAG + KG")

if api_key and build and urls:
    os.environ["GOOGLE_API_KEY"] = api_key
    google.auth._default._CLOUD_SDK_CREDENTIALS_WARNING = False
    with st.spinner("🔍 Loading documents & building Knowledge Graph..."):
        try:
            # --- Load Documents ---
            docs = sum([WebBaseLoader([u]).load() for u in urls], [])
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)

            # --- Vector Store ---
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = Chroma.from_documents(splits, embedding=embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 8})

            # --- spaCy NER + Knowledge Graph ---
            nlp = spacy.load("en_core_web_sm")
            graph = NetworkxEntityGraph()
            G = nx.Graph()

            for d in splits:
                doc = nlp(d.page_content)
                ents = {ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "PRODUCT", "GPE", "NORP", "EVENT"]}
                for e in ents:
                    G.add_node(e)
                    if not G.has_edge(e, "Concept"):
                        G.add_edge(e, "Concept", relation="related_to")
            graph.graph = G

            # # --- KG Search ---
            # def kg_search(q):
            #     q_tokens = [t.text.lower() for t in nlp(q)]
            #     hits = [n for n in G.nodes if any(tok in n.lower() for tok in q_tokens)]
            #     if hits:
            #         rels = [(a, b) for a, b in G.edges() if a in hits or b in hits]
            #         return f"Entities: {hits[:3]} | Relations: {rels[:5]}"
            #     return "No related entities found."

            # --- Memory + Tools + Agent ---
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            memory = ConversationBufferMemory(return_messages=True)
            tools = [
                Tool(
                    "RAG Retriever",
                    lambda q: "\n".join([d.page_content for d in retriever.get_relevant_documents(q)]),
                    "Use for factual answers from documents."
                ),
                Tool(
                    "Chat Memory",
                    lambda q: "\n".join([m.content for m in memory.chat_memory.messages]),
                    "Use when user refers to previous chats."
                ),
                Tool(
                    "Knowledge Graph Search",
                    lambda q: kg_search(q, nlp, G),
                    "Use for entity, people, or relationship questions."
                )
            ]
            agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description",
                                     verbose=False, handle_parsing_errors=True)
            agent.memory = memory

            st.session_state["agent"] = agent
            st.success("✅ RAG + KG successfully initialized!")

        except Exception as e:
            st.error(f"❌ Error while building: {e}")

# --- Chat Interface ---
if "agent" in st.session_state:
    agent = st.session_state["agent"]
    st.divider()
    st.subheader("💬 Chat with Agentic RAG")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask your question here...", placeholder="e.g., What are cookies on a website?")

    if st.button("🚀 Send"):
        if user_input:
            with st.spinner("🤔 Thinking..."):
                prompt = f"""
You are an Agentic RAG assistant with [RAG Retriever, Chat Memory, Knowledge Graph Search].
Select tools automatically. Prefer KG for entities, RAG for factuals, Memory for past context.
Question: {user_input}
"""
                try:
                    ans = agent.run(prompt)
                    st.session_state.chat_history.append(("user", user_input))
                    st.session_state.chat_history.append(("bot", ans))
                    agent.memory.chat_memory.add_user_message(user_input)
                    agent.memory.chat_memory.add_ai_message(ans)
                except Exception as e:
                    st.session_state.chat_history.append(("bot", f"⚠️ Error: {e}"))

    # Display conversation
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"🧑 **You:** {msg}")
        else:
            st.markdown(f"🤖 **Agent:** {msg}")
else:
    st.info("👈 Add one or more URLs and click **Build RAG + KG** to begin chatting.")
