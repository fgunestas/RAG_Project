import requests
from dotenv import load_dotenv
import os

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def web_search(query: str, max_results: int = 5):
    print(">> web searching...")
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

    # Sonuçları basitçe toparla
    results = []
    for item in data.get("results", []):
        title = item.get("title", "")
        link = item.get("url", "")
        snippet = item.get("content", "")
        results.append(f"{title}\n{snippet}\n{link}\n")

    return "\n\n".join(results)