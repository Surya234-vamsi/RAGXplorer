# ---pages/1_Naive_RAG.py---
import os
import google.auth
import streamlit as st
from langchain_core.prompts import PromptTemplate

# Set default User-Agent
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.core_utils import join_splits

# --- PAGE CONFIG ---
st.set_page_config(page_title="Naive RAG App", page_icon="🧠", layout="centered")
st.title("🧠 Naive RAG with Gemini + LangChain")
st.caption("Ask questions about any website using Gemini 2.5 Flash")

# --- USER INPUTS ---
api_key = st.text_input("🔑 Enter your Gemini API key", type="password")
url = st.text_input("🌐 Enter a website URL to load", placeholder="https://example.com")

# # --- LOAD ENV VARIABLES ---
# load_dotenv()  # Loads variables from .env file
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

if api_key and url:
    try:
        # --- SETUP GOOGLE API KEY ---
        os.environ["GOOGLE_API_KEY"] = api_key
        google.auth._default._CLOUD_SDK_CREDENTIALS_WARNING = False

        # --- LOAD WEBSITE ---
        with st.spinner("📡 Scraping website..."):
            loader = WebBaseLoader(web_paths=[url])
            docs = loader.load()

        # --- SPLIT TEXT ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)

        # --- CREATE VECTOR STORE ---
        with st.spinner("⚙️ Creating embeddings and vector DB..."):
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
            retriever = vectorstore.as_retriever()

        # --- PROMPT + LLM SETUP ---
        prompt = PromptTemplate.from_template("""
        You are a helpful assistant that answers questions based on provided context.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer the question based on the context provided. If the answer cannot be found 
        in the context, say "I don't have enough information to answer that."
        
        Answer:""")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        # # --- JOIN CONTEXT FUNCTION ---
        # def join_splits(docs):
        #     return "\n".join(doc.page_content for doc in docs)

        # --- RAG CHAIN ---
        rag_chain = (
            {"context": retriever | join_splits, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        st.success("✅ Website loaded and model ready!")
        st.markdown("---")

        # --- CHAT SECTION ---
        st.subheader("💬 Ask a Question")
        user_query = st.text_input("Type your question below 👇")

        if user_query:
            with st.spinner("🧠 Thinking..."):
                try:
                    response = rag_chain.invoke(user_query)
                    st.markdown("### 💡 Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

    except Exception as e:
        st.error(f"❌ Failed to initialize: {e}")

else:
    st.info("👆 Please enter both your Gemini API key and a valid website URL to begin.")
