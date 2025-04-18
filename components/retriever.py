from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import os

def get_vectorstore():
    pdfs = []
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".pdf"):
                pdfs.append(os.path.join(root, file))

    docs = []
    for pdf in pdfs:
        loader = PyMuPDFLoader(pdf)
        temp = loader.load()
        docs.extend(temp)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

    vector = embedding_model.embed_query(chunks[0].page_content)
    vector_store = FAISS.from_documents(chunks, embedding_model)




    return vector_store
