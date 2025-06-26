from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

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
            "result": [WEB_SEARCH_MANIFEST]
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

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
