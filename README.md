<h1 align="center">🧠 LangGraph Tabanlı RAG Ajanı</h1>
<p align="center">
    Kendi PDF verinizle çalışan, LangChain + LangGraph + Ollama tabanlı güçlü bir bilgi sorgulama altyapısı. <br>
    Agent destekli RAG akışı, yerel LLM entegrasyonu ve belge değerlendirme (grading) özellikleriyle uçtan uca çözümler.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LangChain-0.1.x-blue?logo=python">
  <img src="https://img.shields.io/badge/LangGraph-Enabled-9cf?logo=graphql">
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-success?logo=chatbot">
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-brightgreen">
  <img src="https://img.shields.io/badge/PDF-Support-lightgrey?logo=adobeacrobatreader">
</p>

---

📂 **Veri Kaynağı**: `data/` klasöründeki PDF dosyaları  
🧩 **İşlem Akışı**: PDF → Chunk → Embed → FAISS → Retrieve → Grade → Generate  
🖥️ **Tamamen Lokal**: İnternet bağlantısı veya API key gerekmez  
🧠 **LLM Modeli**: `llama3` (Ollama)  
🔍 **Embedding**: `nomic-embed-text` (Ollama)  
🔧 **Graph Mantığı**: `LangGraph` ile yönlendirilmiş RAG & Tool-Calling Ajanı  
🧪 **Notebook**: Jupyter üzerinden adım adım gösterim

---

> 🎯 **Hedef**: Belgelerden anlamlı bilgi çıkaran, esnek ve geliştirilebilir bir bilgi sorgulama sistemi kurmak.


