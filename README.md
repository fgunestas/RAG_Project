<h1 align="center">🧠 Smart RAG Agent Powered by LangGraph</h1>
<p align="center">
    An intelligent information retrieval system powered by LangChain + LangGraph + Ollama, working with your local PDF data. <br>
    Includes web search integration, multilingual support, and document grading for a complete end-to-end solution.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LangChain-0.1.x-blue?logo=python">
  <img src="https://img.shields.io/badge/LangGraph-Router%20+%20Grader%20Graph-9cf?logo=graphql">
  <img src="https://img.shields.io/badge/Ollama-Mistral%207B-success?logo=chatbot">
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-brightgreen">
  <img src="https://img.shields.io/badge/WebSearch-Travil-red?logo=google">
  <img src="https://img.shields.io/badge/MultiLang-Supported-lightgrey?logo=translate">
</p>

---

### 🚀 Features

📂 **Data Source**: PDF files located in the `data/` directory  
🧩 **Processing Flow**: PDF → Chunk → Embed → FAISS → Retrieve → Grade → Route → Generate  
🖥️ **Fully Local**: Can run offline without internet (web search is optional)  
🔎 **Retriever Evaluation**: Document relevance is evaluated by a `grader` node  
🌍 **Web Search Integration**: External knowledge fetching via `Travil` API + `LangChain MCP`  
🧠 **LLM Model**: `mistral:7b-instruct` via Ollama  
🔤 **Language Matching**: LLM responds in the same language as the user question  
📚 **Embedding**: `nomic-embed-text` (via Ollama)  
🧠 **Graph Architecture**:  


![Test Image 1](https://github.com/fgunestas/RAG_Project/blob/main/graph.png)

📓 **Demo Notebook**: 📖 [Jupyter Notebooks](https://nbviewer.org/github/fgunestas/RAG_Project/blob/main/test_notebook.ipynb)

---

### 🎯 Purpose

To build a **flexible, extensible, and fully local RAG pipeline** that can extract meaningful insights from documents, access real-time web data when needed, and support multilingual interactions.

---
## 🚧 Build & Run

### 1️⃣ Requirements

#### 🐍 Python (>=3.10) and pip

> If Python is not installed, download the appropriate version for your system from [python.org](https://www.python.org/downloads/).
```bash
python --version
# Must be Python 3.10 or later
```

#### 🧠 Ollama (for local LLM)
 Ollama must be installed to run local models like Mistral 7B.\
>[➤Download Ollama](https://ollama.com/download)

#### 🌐 Travil API Key (for Web Search)
If you want to enable web search fallback, you need a Travil API key.\
Visit [travil.dev](travil.dev)\
Sign up and create an API key

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/fgunestas/RAG_Project.git
cd RAG_Project
```
### 3️⃣ Create a .env File
Create a .env file in the root directory with the following content:
```bash
TRAVIL_API_KEY=your_travil_api_key_here
```
### 5️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
7️⃣ Run the MCP server
```bash
python mcp_server.py
```