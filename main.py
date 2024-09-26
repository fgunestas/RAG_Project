import os
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
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,pipeline

model_name = 'HuggingFaceH4/zephyr-7b-beta'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)



text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:c

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

 """
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)



def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def load_documents():
    loader=WebBaseLoader("https://python.langchain.com/docs/get_started/introduction")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
    return splitter.split_documents(documents)

def load_embeddings(documents):
    db = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'))
    return db.as_retriever(search_type='similarity', search_kwargs={'k': 3})

def generate_response(retriever, query):
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return  chain.invoke(query)

def query(query):
    documents = load_documents()
    retriever = load_embeddings(documents)
    response =  generate_response(retriever, query)
    return response
def start():
    ask()

def ask():
    while True:
        user_input = input("Q:")
        response= query(user_input)
        print("A:", response)

if __name__=="__main__":
    start()