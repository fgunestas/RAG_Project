import os
from re import search

import bs4
from dotenv import load_dotenv

import warnings

warnings.filterwarnings("ignore")

load_dotenv()

import torch
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,pipeline
from langchain.load import dumps, loads
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader


loader = PyMuPDFLoader(r"*.pdf")
loader.load()

model_name = "llama3.2:3b"

llm = ChatOllama(model=model_name, base_url="http://localhost:8080")



embeddings= OllamaEmbeddings(model="nomic-embed-text",base_url='http://localhost:8080')
db_name="test"
vector_store=FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)

retriever= vector_store.as_retriever(search_type="similarity", search_kwargs={'k':5})







multi_query_template = PromptTemplate(
    input_variables=["question"],
    template="""
1. Expert Opinion: What would domain experts say about the question "{question}"?
2. Historical Context: How has the topic of "{question}" evolved over time?
3. Practical Application: What are some real-world examples or case studies related to the question "{question}"?
4. Contradictory Viewpoints: Are there any counterarguments or opposing perspectives regarding the question "{question}"?
5. Ethical Considerations: What are the ethical issues surrounding the question "{question}"?
6. Future Outlook: What might be the future trends or developments related to the question "{question}"?
"""
)

rag_template="""Answer the following question based on this context: {context} 
Question: {question}"""

rag_prompt=ChatPromptTemplate.from_template(rag_template)




def load_documents():

    loader=WebBaseLoader(web_paths=["https://blog.langchain.dev/deconstructing-rag/"],
                         bs_kwargs=dict(
                             parse_only=bs4.SoupStrainer(
                                 class_=("article-header section", "article-header__content", "article-header__footer")
                             )

                         ))

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=30)
    return splitter.split_documents(documents)

def load_embeddings(documents):
    db = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'))
    return db.as_retriever(search_type='similarity', search_kwargs={'k': 3})


def generate_multi_queries():
    return(
        multi_query_template
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
def get_unique_union(documents: list[list]):
    flattened_docs=[dumps(doc) for sublist in documents for doc in sublist]
    unique_docs=list(set(flattened_docs))
    return  [loads(doc) for doc in unique_docs]



def query(query):
    documents = load_documents()
    retriever = load_embeddings(documents)
    retrieval_chain=generate_multi_queries() | retriever.map() | get_unique_union



    chain_for_generating=(
        {"context": retrieval_chain, "question":RunnablePassthrough()}
        |rag_prompt
        |llm
        |StrOutputParser()
    )
    return chain_for_generating.invoke(query)