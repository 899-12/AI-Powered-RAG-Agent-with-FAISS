import os
import tempfile
import streamlit as st
from datetime import datetime
import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

# --- Load API Keys from Environment Variables ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")

# --- Initialize FAISS Vector Store ---
INDEX_DIM = 768  # Match embedding dimension
index = faiss.IndexFlatL2(INDEX_DIM)
vector_store = {}

# --- Initialize Embedding Model ---
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Streamlit Setup ---
st.title("🔍 AI-Powered RAG Agent with FAISS")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Settings")
if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.rag_enabled = st.sidebar.toggle("Enable RAG Mode", value=True)
st.session_state.use_web_search = st.sidebar.checkbox("Enable Web Search", value=False)

# --- Function to Split Documents ---
def split_texts(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return [Document(page_content=chunk.page_content, metadata=chunk.metadata) for chunk in splitter.split_documents(documents)]

# --- Function to Process PDF ---
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        documents = PyPDFLoader(tmp_file.name).load()
    
    for doc in documents:
        doc.metadata.update({"source": uploaded_file.name, "timestamp": datetime.now().isoformat()})
    
    return split_texts(documents)

# --- Function to Process Web Page ---
def process_web(url):
    try:
        documents = WebBaseLoader(url).load()
        for doc in documents:
            doc.metadata.update({"source": url, "timestamp": datetime.now().isoformat()})
        return split_texts(documents)
    except Exception as e:
        st.error(f"Error processing web page: {str(e)}")
        return []

# --- Upload Data ---
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
web_url = st.sidebar.text_input("Enter a Web URL")

if uploaded_file:
    docs = process_pdf(uploaded_file)
    if docs:
        embeddings = embedding_model.embed_documents([d.page_content for d in docs])
        embeddings = np.array(embeddings).astype("float32")
        index.add(embeddings)
        vector_store[uploaded_file.name] = docs

if web_url:
    docs = process_web(web_url)
    if docs:
        embeddings = embedding_model.embed_documents([d.page_content for d in docs])
        embeddings = np.array(embeddings).astype("float32")
        index.add(embeddings)
        vector_store[web_url] = docs

# --- Retrieve Documents ---
def retrieve_documents(query):
    query_embedding = np.array(embedding_model.embed_query(query)).astype("float32").reshape(1, -1)
    D, I = index.search(query_embedding, k=5)
    retrieved_docs = [vector_store[key][i] for key in vector_store for i in I[0] if i < len(vector_store[key])]
    return [doc.page_content for doc in retrieved_docs]

# --- AI Agents ---
def get_rag_agent():
    return Agent(name="RAG Agent", model=Ollama(id=OLLAMA_MODEL), instructions="Answer using retrieved data.")

def get_web_search_agent():
    return Agent(name="Web Search", model=Gemini(id="gemini-2.0-flash-exp"), tools=[DuckDuckGoTools()])

# --- Process Query ---
def process_query(query):
    st.session_state.history.append({"role": "user", "content": query})
    
    context = ""
    if st.session_state.rag_enabled:
        retrieved_docs = retrieve_documents(query)
        context = "\n\n".join(retrieved_docs)
    
    if st.session_state.use_web_search and not context:
        with st.spinner("Searching the web..."):
            web_results = get_web_search_agent().run(query).content
            context = f"Web Search Results:\n{web_results}" if web_results else ""
    
    with st.spinner("Generating response..."):
        response = get_rag_agent().run(f"Context: {context}\n\nQuestion: {query}").content
        st.session_state.history.append({"role": "assistant", "content": response})
        return response

# --- Chat Interface ---
prompt = st.chat_input("Ask a question...")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    
    response = process_query(prompt)
    with st.chat_message("assistant"):
        st.write(response)
