from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from .retriever import get_vectorstore

llm = ChatOllama(model="llama3.2:3b", base_url="http://localhost:8080")

def rag_query(query: str):
    retriever = get_vectorstore().as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)