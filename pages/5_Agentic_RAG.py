# ---pages/5_Agentic_RAG.py---

import os
import google.auth
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent

# --- PAGE CONFIG ---
st.set_page_config(page_title="Agentic RAG", page_icon="🧠", layout="centered")
st.title("🧠 Agentic RAG Chatbot")
st.caption("Autonomous reasoning assistant — decides when to retrieve, recall, or infer using Gemini 2.5 Flash")

# --- USER INPUTS ---
api_key = st.text_input("🔑 Enter your Gemini API key", type="password")
urls_input = st.text_area(
    "🌐 Enter one or more website URLs (one per line)",
    "https://www.lorique.net/posts/web/why-this-site-doesnt-use-cookies/\n"
    "https://support.mozilla.org/en-US/kb/clear-cookies-and-site-data-firefox",
)

# --- MAIN LOGIC ---
if api_key and urls_input:
    os.environ["GOOGLE_API_KEY"] = api_key
    google.auth._default._CLOUD_SDK_CREDENTIALS_WARNING = False
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]

    try:
        with st.spinner("🔍 Loading and preparing documents..."):
            # --- Load documents ---
            docs = sum([WebBaseLoader([u]).load() for u in urls], [])
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # --- Create embeddings and vectorstore ---
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = Chroma.from_documents(splits, embedding=embeddings)
            vector_retriever = db.as_retriever(search_kwargs={"k": 8})

            # --- BM25 retriever (lexical) ---
            bm25_retriever = BM25Retriever.from_documents(splits)

            # --- Hybrid retriever (combine both) ---
            def hybrid_retrieve(query):
                bm25_docs = bm25_retriever.get_relevant_documents(query)
                vector_docs = vector_retriever.get_relevant_documents(query)
                # merge and deduplicate by content
                all_texts = {}
                for d in bm25_docs + vector_docs:
                    all_texts[d.page_content] = d
                return list(all_texts.values())

            # --- LLM ---
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

            # --- Memory ---
            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(return_messages=True)
            memory = st.session_state.memory

            # --- Define tools ---
            tools = [
                Tool(
                    name="Hybrid Retriever",
                    func=lambda q: "\n".join([d.page_content for d in hybrid_retrieve(q)]),
                    description="Use this when the question needs factual/document-based information."
                ),
                Tool(
                    name="Chat Memory",
                    func=lambda q: "\n".join([m.content for m in memory.chat_memory.messages]),
                    description="Use this when referring to previous chat or context."
                ),
            ]

            # --- Initialize agent ---
            agent = initialize_agent(
                tools,
                llm,
                agent_type="zero-shot-react-description",
                verbose=False,
                handle_parsing_errors=True,
                memory=memory
            )

        st.success("✅ Documents loaded and Agent initialized!")
        st.markdown("---")

        # --- CHAT SECTION ---
        st.subheader("💬 Ask a Question")
        q = st.text_input("Your question:")

        if q:
            with st.spinner("🤔 Agent thinking..."):
                prompt = f"""
You are an intelligent Agentic RAG assistant.
You have tools: [Hybrid Retriever, Chat Memory].
Decide autonomously how to answer the question using these tools or your own knowledge.
Be concise and confident.

Question: {q}
"""
                try:
                    ans = agent.run(prompt)
                except Exception as e:
                    ans = f"⚠️ Error during reasoning: {str(e)}"

                st.markdown("### 💡 Agentic Answer")
                st.write(ans)

                # Save conversation
                memory.chat_memory.add_user_message(q)
                memory.chat_memory.add_ai_message(ans)

        # --- SHOW MEMORY ---
        if st.session_state.memory.chat_memory.messages:
            st.markdown("---")
            st.subheader("🧠 Conversation Memory")
            for m in reversed(st.session_state.memory.chat_memory.messages[-10:]):
                if m.type == "human":
                    st.markdown(f"**🧑 You:** {m.content}")
                else:
                    st.markdown(f"**🤖 Agent:** {m.content}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("👆 Please enter your Gemini API key and at least one website URL to start chatting.")
