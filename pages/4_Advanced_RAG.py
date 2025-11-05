# ---pages/4_Advanced_RAG.py---

import os
import google.auth
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

# Set default User-Agent
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory


# --- PAGE CONFIG ---
st.set_page_config(page_title="Advanced Hybrid RAG", page_icon="🤖", layout="centered")
st.title("🤖 Advanced Hybrid RAG Chatbot")
st.caption("Auto-selects between RAG, LLM, and Memory using Gemini 2.5 Flash + LangChain")


# --- UNIVERSAL HYBRID RETRIEVER ---
class HybridRetriever:
    def __init__(self, retriever1, retriever2, top_k=5):
        self.retriever1 = retriever1
        self.retriever2 = retriever2
        self.top_k = top_k

    def get_relevant_documents(self, query):
        """Supports both old and new retriever APIs."""
        def get_docs(r):
            if hasattr(r, "get_relevant_documents"):
                return r.get_relevant_documents(query)
            elif hasattr(r, "invoke"):
                return r.invoke(query)
            else:
                return []

        docs1 = get_docs(self.retriever1)
        docs2 = get_docs(self.retriever2)
        unique = {d.page_content: d for d in docs1 + docs2}
        return list(unique.values())[:self.top_k]


# --- USER INPUTS ---
api_key = st.text_input("🔑 Enter your Gemini API key", type="password")
urls_input = st.text_area(
    "🌐 Enter one or more website URLs (one per line)",
    "https://www.lorique.net/posts/web/why-this-site-doesnt-use-cookies/\nhttps://support.mozilla.org/en-US/kb/clear-cookies-and-site-data-firefox",
)

if api_key and urls_input:
    os.environ["GOOGLE_API_KEY"] = api_key
    google.auth._default._CLOUD_SDK_CREDENTIALS_WARNING = False
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]

    try:
        with st.spinner("🔍 Loading documents and building vector database..."):
            # --- LOAD DOCS ---
            docs = sum([WebBaseLoader([u]).load() for u in urls], [])

            # --- SPLIT ---
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # --- VECTOR STORE ---
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = Chroma.from_documents(splits, embedding=embeddings)
            vector_retriever = db.as_retriever(search_kwargs={"k": 5})

            # --- BM25 ---
            bm25_retriever = BM25Retriever.from_documents(splits)

            # --- HYBRID RETRIEVER ---
            advanced_retriever = HybridRetriever(vector_retriever, bm25_retriever, top_k=10)

            # --- LLM + MEMORY ---
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(return_messages=True)
            memory = st.session_state.memory

        st.success("✅ Data successfully loaded and retriever initialized!")
        st.markdown("---")

        # --- CHAT SECTION ---
        st.subheader("💬 Ask a Question")
        q = st.text_input("Your question:")

        if q:
            with st.spinner("🧠 Thinking..."):
                refined_q = llm.invoke(f"Refine this query for retrieval and reasoning:\n{q}").content

                docs_context = advanced_retriever.get_relevant_documents(refined_q)
                context = "\n".join([d.page_content for d in docs_context])

                history = "\n".join([
                    f"User: {m.content}" if m.type == "human" else f"Bot: {m.content}"
                    for m in memory.chat_memory.messages
                ])

                judge_prompt = f"""
You are a reasoning controller for a hybrid RAG system.
Decide which source to use:
1. "RAG" → if context clearly contains relevant info
2. "LLM" → if general question
3. "MEMORY" → if refers to chat history
Question: {q}
Context: {context[:700]}
History: {history[-700:]}
Only output one word: RAG, LLM, or MEMORY.
"""
                decision = llm.invoke(judge_prompt).content.strip().upper()
                st.markdown(f"### 🤖 Decision: `{decision}`")

                if "RAG" in decision:
                    prompt = f"""Use this context to answer:
{context}
Question: {q}"""
                    src = "📘 Retrieved Docs"
                elif "MEMORY" in decision:
                    prompt = f"""Use the chat history:
{history}
Question: {q}"""
                    src = "🧠 Memory Context"
                else:
                    prompt = f"Answer this using general knowledge:\n{q}"
                    src = "🌐 LLM Knowledge"

                ans = llm.invoke(prompt).content

                memory.chat_memory.add_user_message(q)
                memory.chat_memory.add_ai_message(ans)

                st.markdown("### 💡 Answer")
                st.write(ans)
                st.caption(f"Source: {src}")

        if st.session_state.memory.chat_memory.messages:
            st.markdown("---")
            st.subheader("🧠 Conversation History")
            for m in reversed(st.session_state.memory.chat_memory.messages[-10:]):
                if m.type == "human":
                    st.markdown(f"**🧑 You:** {m.content}")
                else:
                    st.markdown(f"**🤖 Bot:** {m.content}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

else:
    st.info("👆 Enter your Gemini API key and at least one website URL to start chatting.")
