# ---pages/2_Conversational_RAG.py---

import os
import google.auth
import streamlit as st
from collections import deque
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



# --- PAGE CONFIG ---
st.set_page_config(page_title="Conversational RAG", page_icon="💬", layout="centered")
st.title("💬 Conversational RAG with Memory")
st.caption("Ask questions about any website and retain conversation context.")

# --- USER INPUTS ---
api_key = st.text_input("🔑 Enter your Gemini API key", type="password")
url = st.text_input("🌐 Enter a website URL", placeholder="https://example.com")

# # --- LOAD ENV VARIABLES ---
# load_dotenv()  # Loads variables from .env file
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

if api_key and url:
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        google.auth._default._CLOUD_SDK_CREDENTIALS_WARNING = False

        with st.spinner("📡 Loading website and preparing data..."):
            # --- LOAD WEBSITE ---
            loader = WebBaseLoader([url])
            docs = loader.load()

            # --- SPLIT DOCS ---
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)

            # --- EMBEDDINGS + CHROMA DB ---
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = Chroma.from_documents(splits, embedding=embeddings)
            retriever = db.as_retriever()

        # --- LLM + PROMPT ---
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

        prompt = PromptTemplate.from_template("""
        You are a helpful assistant that answers questions using both
        the retrieved context and the previous chat history.

        Chat history:
        {history}

        Context from documents:
        {context}

        User question:
        {question}

        Answer clearly and concisely:
        """)

        # --- RAG CHAIN ---
        rag_chain = (
            {
                "context": RunnablePassthrough() | (lambda x: retriever.invoke(x["question"])),
                "question": RunnablePassthrough() | (lambda x: x["question"]),
                "history": RunnablePassthrough() | (lambda x: x["history"]),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        st.success("✅ Website loaded and ready for chat!")

        # --- MEMORY SETUP ---
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = deque(maxlen=3)

        st.markdown("---")
        st.subheader("💬 Ask a question")

        user_q = st.text_input("Type your question below 👇")

        if user_q:
            # Combine past memory
            history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.conversation_memory])

            with st.spinner("🧠 Thinking..."):
                try:
                    response = rag_chain.invoke({
                        "question": user_q,
                        "context": user_q,
                        "history": history_text
                    })
                    st.markdown("### 💡 Answer:")
                    # st.write(response)

                    # Save Q&A to memory
                    st.session_state.conversation_memory.append((user_q, response))

                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

        if st.session_state.conversation_memory:
            st.markdown("---")
            st.subheader("🧠 Conversation Memory")
            for i, (q, a) in enumerate(st.session_state.conversation_memory):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a}")

    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")

else:
    st.info("👆 Please enter both your Gemini API key and a valid website URL to begin.")
