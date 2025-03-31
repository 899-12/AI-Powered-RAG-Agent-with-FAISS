# AI-Powered RAG Agent with FAISS

## 🔍 Overview
This project is an **AI-powered Retrieval-Augmented Generation (RAG) system** that uses **FAISS** for efficient document retrieval and integrates **LLMs** for intelligent query responses. It supports PDF and web page ingestion, embedding-based search, and optional web search augmentation.

## 🚀 Features
- **Upload & Process PDFs**: Extracts and embeds text from PDFs.
- **Fetch & Process Web Pages**: Retrieves and embeds content from web pages.
- **FAISS-based Document Retrieval**: Efficient similarity search.
- **RAG Mode**: Uses retrieved data for more informed answers.
- **Web Search Option**: Enables external knowledge search.
- **Conversational Interface**: Powered by Streamlit.

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo.git](https://github.com/899-12/AI-Powered-RAG-Agent-with-FAISS.git)
   cd your-repo
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Set up environment variables (create a `.env` file or export manually):
   ```bash
   export GOOGLE_API_KEY="your-google-api-key"
   export OLLAMA_MODEL="deepseek-r1:1.5b"
   ```

## 📌 Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

## 🏗️ How It Works
1. **Document Upload & Processing:**
   - PDFs are extracted using `PyPDFLoader`.
   - Web pages are processed using `WebBaseLoader`.
   - Text is split into chunks using `RecursiveCharacterTextSplitter`.
2. **Embedding & Indexing:**
   - Text is embedded using Google Generative AI Embeddings (`models/embedding-001`).
   - Stored in a FAISS index for fast retrieval.
3. **Retrieval-Augmented Generation (RAG):**
   - User queries are embedded and searched in FAISS.
   - Top retrieved documents are passed to an LLM (`Ollama`).
4. **Web Search Augmentation (Optional):**
   - If enabled, external web search via `DuckDuckGoTools` provides additional context.
5. **Chat Interface:**
   - Users interact through a **Streamlit chat UI**.
   
## 🏗️ Dependencies
- `streamlit`
- `faiss`
- `numpy`
- `langchain`
- `agno`
- `duckduckgo-search`

## 🤝 Contributing
Feel free to open issues or submit PRs if you have improvements!



---
🎯 **Built with AI, RAG, and Love ❤️**

