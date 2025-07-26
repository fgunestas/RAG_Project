import requests
from dotenv import load_dotenv
import os
from components.retriever import get_vectorstore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from mcp.server.fastmcp import FastMCP

load_dotenv()


vector_store= get_vectorstore()
retriever = vector_store.as_retriever(search_type="similarity", k=5)
llm = ChatOllama(model="mistral:7b", base_url="http://localhost:11434")

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Always reply in the same language as the user question.

Conversation History:
{chat_history}

User Question:
{query}

Relevant Context:
{context}

Answer based only on the context above. If the context is insufficient, politely say so.
""")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

mcp = FastMCP("MyRAGServer")

@mcp.tool()
def web_search(query: str) -> dict:
    """Performs a web search using Tavily API."""
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": 5,
        },
    )
    response.raise_for_status()
    return {"rows": response.json()}

@mcp.tool()
def rag_search(query: str, history: str = "") -> dict:
    """Performs RAG-style retrieval and LLM response generation."""
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    result = llm.invoke(rag_prompt.invoke({"query": query, "context": context, "chat_history": history}))
    return {"context": result.content}

if __name__ == "__main__":
    mcp.run(transport="streamable-http")


RAG_MANIFEST = {
    "name": "rag_search",
    "description": "Performs RAG-style semantic document retrieval.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user query to search relevant context."
            }
        },
        "required": ["query"]
    }
}
WEB_SEARCH_MANIFEST = {
  "name": "web_search",
  "description": "Performs a web search using Serper.dev API.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query string."
      }
    },
    "required": ["query"]
  }
}