import requests
import os
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
print("DEBUG - TAVILY_API_KEY:", TAVILY_API_KEY)

response = requests.post(
    "https://api.tavily.com/search",
    json={
        "api_key": TAVILY_API_KEY,
        "query": "beşiktaş fenerbahçe basketbol final seriisinde son durum ne",
        "max_results": 5
    }
)

print("Status Code:", response.status_code)
print(response.json())