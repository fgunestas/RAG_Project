from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
import os
from components.retriever import get_vectorstore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = Flask(__name__)

vector_store= get_vectorstore()
retreiver = vector_store.as_retriever(search_type="similarity", k=5)
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
llm = ChatOllama(
    model="mistral:7b-instruct",
    base_url="http://localhost:11434"
)

@app.route("/mcp", methods=["POST"])
def handle_mcp():
    req = request.json
    method = req.get("method")
    params = req.get("params", {})
    request_id = req.get("id", 1)

    if method == "mcp.get_tool_manifest":
        return jsonify({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": [WEB_SEARCH_MANIFEST, RAG_MANIFEST]
        })


    elif method == "web_search":

        search_query = params.get("query", "")

        response = requests.post(

            "https://api.tavily.com/search",

            json={

                "api_key": TAVILY_API_KEY,

                "query": search_query,

                "max_results": 5

            }

        )

        response.raise_for_status()

        results = response.json()

        return jsonify({

            "jsonrpc": "2.0",

            "id": request_id,

            "result": {"rows": results}

        })
    elif method == "rag_search":
        query = params.get("query", "")
        docs = retreiver.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        history = params.get("history", "")
        response = llm.invoke(rag_prompt.invoke({"query": query, "context": context, "chat_history": history}))
        result = response.content

        return jsonify({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"context": result}
        })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
