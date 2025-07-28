<h1 align="center">🧠 Smart RAG Agent Powered by LangGraph</h1>
<p align="center">
    An intelligent multi-agent RAG system using LangChain, LangGraph, and Ollama with tool-calling support. <br>
    Automatically chooses between web search, internal document retrieval, or base LLM responses with no need for a manual router node.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LangChain-0.1.x-blue?logo=python">
  <img src="https://img.shields.io/badge/LangGraph-Tool%20Calling%20Graph-9cf?logo=graphql">
  <img src="https://img.shields.io/badge/Ollama-Mistral%207B%20v0.3-success?logo=chatbot">
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-brightgreen">
  <img src="https://img.shields.io/badge/WebSearch-Travil-red?logo=google">
  <img src="https://img.shields.io/badge/Tool%20Calling-Auto-dodgerblue">
  <img src="https://img.shields.io/badge/MultiLang-Supported-lightgrey?logo=translate">
</p>

---

### 🚀 Features

📂 **Data Source**: PDF files placed under `data/` directory  
🧠 **Automatic Tool Calling**: LLM determines whether to call internal retrieval (`rag_search`), web search (`web_search`), or answer directly  
🧩 **MCP-Driven Planning**: MCP handles routing logic based on system state and planner LLM output  
🧠 **LLM Model**: `mistral:7b` (with tool-calling) via Ollama  
🔗 **Function Calling Compatible**: Uses [Mistral 7B v0.3](https://ollama.com/library/mistral) with Ollama’s raw mode  
🌍 **Web Search Integration**: External fallback via `Travil API`  
📚 **Embedding Model**: `nomic-embed-text` (Ollama)  
💬 **Language Awareness**: Answers in the language of the user prompt  
📊 **Graph Logic**: Tool-calling-based LangGraph structure — no manual routing required  
📓 **Notebook Demo**: [▶️ Test Notebook](https://nbviewer.org/github/fgunestas/RAG_Project/blob/main/test_notebook.ipynb)

---

### 🧠 Agent Structure


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