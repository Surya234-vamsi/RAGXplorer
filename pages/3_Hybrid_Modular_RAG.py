# ---pages/3_Hybrid_Modular_Rag.py---

import os
import langchain
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import google.auth
from utils.core_utils import ConversationBufferMemory


# --- STREAMLIT PAGE SETUP ---
st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Hybrid / Modular RAG Chatbot")
st.caption("Chat using both retriever-based context and conversational memory (Gemini + LangChain).")

# --- USER INPUTS ---
api_key = st.text_input("🔑 Enter your Gemini API key", type="password")
url = st.text_input("🌐 Enter website URL", placeholder="https://example.com")

# # --- LOAD ENV VARIABLES ---
# load_dotenv()  # Loads variables from .env file
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

if api_key and url:
    try:
        # --- ENV SETUP ---
        os.environ["GOOGLE_API_KEY"] = api_key
        google.auth._default._CLOUD_SDK_CREDENTIALS_WARNING = False

        with st.spinner("🔍 Loading website and preparing embeddings..."):
            # --- LOAD DOCUMENTS ---
            docs = WebBaseLoader([url]).load()

            # --- SPLIT TEXT ---
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)

            # --- VECTOR STORE ---
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = Chroma.from_documents(splits, embedding=embeddings)
            retriever = db.as_retriever()

            # --- LLM & MEMORY ---
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
            memory = ConversationBufferMemory(return_messages=True)

        st.success("✅ Website data loaded and vector store initialized!")
        st.markdown("---")

        # --- CHAT INTERFACE ---
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            st.session_state.memory = memory

        st.subheader("💬 Ask your question")
        user_q = st.text_input("Your question:")

        if user_q:
            with st.spinner("🤔 Thinking..."):
                # Get memory text
                history = "\n".join([
                    f"User: {m.content}" if m.type == "human" else f"Bot: {m.content}"
                    for m in st.session_state.memory.chat_memory.messages
                ])

                # Get relevant chunks
                docs_context = retriever.invoke(user_q)
                context = "\n".join([d.page_content for d in docs_context])

                # Combine sources
                full_input = f"""Use both sources below to answer helpfully.
Chat history:
{history}

Website info:
{context}

Question: {user_q}
"""

                # Get answer
                ans = llm.invoke(full_input).content

                # Display
                st.markdown("### 💡 Answer:")
                # st.write(ans)

                # Update memory
                st.session_state.memory.chat_memory.add_user_message(user_q)
                st.session_state.memory.chat_memory.add_ai_message(ans)
                st.session_state.chat_history.append(("🧑‍💻", user_q))
                st.session_state.chat_history.append(("🤖", ans))

        # --- SHOW CHAT HISTORY ---
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("🧠 Conversation History")
            for role, msg in reversed(st.session_state.chat_history[-10:]):
                st.markdown(f"**{role}:** {msg}")

    except Exception as e:
        st.error(f"⚠️ Error initializing app: {e}")

else:
    st.info("👆 Please enter your Gemini API key and a valid website URL to start.")
