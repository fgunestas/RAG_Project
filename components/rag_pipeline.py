from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
from typing import TypedDict
from components.retriever import get_vectorstore
from components.search_tool import web_search
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests





llm = ChatOllama(
    model="mistral:7b-instruct",
    base_url="http://localhost:11434"
)
vector_store= get_vectorstore()
retreiver = vector_store.as_retriever(search_type="similarity", k=5)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent multilingual assistant. Respond in the same language as the user's question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "User Question:\n{question}\n\nContext:\n{context}\n\nBased on the context, answer accurately and clearly in the user's language. If the context is insufficient, say so.")
])


# State
class AgentState(TypedDict):
    input: str
    llm_output: str
    search_output: str
    context: str
    routing: str
    chat_history: list  # LangChain message list
    planned_input:str
    final_output:str


# === Nodes ===


def llm_rag_node(state: AgentState) -> AgentState:
    print(">> RAG Node çalışıyor...")
    query = state["input"]
    history = state.get("chat_history", [])

    response = requests.post(
        "http://localhost:8000/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "rag_search",
            "params": {"query": query,"chat_history": history},
            "id": 123
        }
    )

    context = response.json()["result"]["context"]
    state["llm_output"] = context

    return state

def llm_node(state: AgentState) -> AgentState:
    question = state["input"]
    context = state.get("search_output", "")
    chat_history = state.get("chat_history", [])



    full_prompt = prompt.invoke({
        "question": question,
        "context": context,
        "chat_history":chat_history

    })

    llm_response = llm.invoke(full_prompt)
    #adding current response to memmory

    state["llm_output"] = llm_response.context
    return state

from langchain.tools import Tool

def rag_tool_func(query: str,history: list) -> str:
    response = requests.post(
        "http://localhost:8000/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "rag_search",
            "params": {"query": query,"chat_history": history},
            "id": 2
        }
    )
    return response.json()["result"]["context"]

rag_tool = Tool.from_function(
    func=rag_tool_func,
    name="rag_search",
    description="Useful for retrieving context from internal documents."
)
web_tool = Tool.from_function(
    func=rag_tool_func,
    name="web_search",
    description=(
    "Use this tool to search the internet for recent or real-time information. "
    "Ideal for answering questions related to current events, news updates, trending topics, weather, or any facts that may have changed over time. "
    "Do not use this for historical, theoretical, or static information."
)
)

def search_node(state: AgentState) -> AgentState:
    question = state["input"]

    response = requests.post(
        "http://localhost:8000/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "web_search",
            "params": {"query": question},
            "id": 2
        }
    )
    context=response.json()["result"]["rows"]

    state["search_output"] = context
    return state

def manager_node(state: AgentState) -> AgentState:
    print("routing...")
    router_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a routing agent in a multi-agent system.\n"
         "Your task is to decide which specialized agent should handle the user's question.\n\n"
         "Decision rules:\n"
         "- route: rag        → If the question is about theoretical or conceptual information related to Kendo\n"
         "                     (e.g., techniques, history, rules, training, terminology)\n"
         "- route: web_search → If the question requires current, factual, or real-time information from the internet\n"
         "                     (e.g., latest sports results, recent news, weather, live events)\n"
         "- route: base_llm   → For all other topics such as general knowledge, personal questions,\n"
         "                     philosophical or humorous queries, social conversation, or chat memory use.\n\n"
         "Only respond with one of the following, no explanation:\n"
         "route: rag\n"
         "route: web_search\n"
         "route: base_llm"
         ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nSearch Output:\n{search_output}")
    ])

    question = state["input"]
    context = state.get("context", "")
    search_output = state.get("search_output", "")
    chat_history = state.get("chat_history", [])


    full_prompt = router_prompt.invoke({
        "question": question,
        "context": context,
        "search_output": search_output,
        "chat_history": chat_history
    })
    response = llm.invoke(full_prompt)
    route = response.content.strip().lower()
    print(route)
    if "rag" in route:
        return "llm_rag"
    elif "web" in route:
        return "web_search"
    else:
        print("else")
        return "base_llm"


def planner_node(state: AgentState) -> AgentState:
    print("planner")

    input_text = state.get("input", "").strip()


    planned_query = input_text.lower()

    state["planned_input"] = planned_query

    return state
def output_node(state: AgentState) -> AgentState:
    print(">> Output Node çalışıyor...")

    question = state.get("input", "")
    agent_output = (
        state.get("llm_output") # RAG or base LLM response
        or state.get("search_output") # web search response
    )


    chat_history = state.get("chat_history", [])


    final_prompt = ChatPromptTemplate.from_template("""
Given the user's question, the previous output generated by an agent (RAG, base model, or web search), 
and the conversation history (if any), generate a final response that is clear, polite, and helpful.

Your response should be in the **same language as the user's question**, unless otherwise specified.

Be sure to directly address the user's question based on the provided information. If the agent's output is already good, you may rephrase or polish it. Otherwise, restructure it or explain better based on context.

User Question:
{question}

Agent Output:
{agent_output}

Conversation History (optional):
{chat_history}

Now write the final response for the user:
""")


    full_prompt = final_prompt.invoke({
        "question": question,
        "agent_output": agent_output,
        "chat_history": chat_history
    })


    final_response = llm.invoke(full_prompt)


    state["final_output"] = final_response.content
    return state


# === Build Graph ===
graph = StateGraph(AgentState)

graph.add_node("planner_llm", planner_node)
graph.add_node("llm_rag", llm_rag_node)
graph.add_node("base_llm", llm_node)
graph.add_node("web_search", search_node)
graph.add_node("output_llm", output_node)



graph.set_entry_point("planner_llm")
graph.add_conditional_edges(
    "planner_llm",
    RunnableLambda(manager_node),
    {
        "llm_rag": "llm_rag",
        "web_search": "web_search",
        "base_llm": "base_llm"
    }
)
graph.add_edge("web_search", "output_llm")
graph.add_edge("llm_rag", "output_llm")
graph.add_edge("base_llm", "output_llm")

graph.set_finish_point("output_llm")


search_agent = graph.compile()
search_agent.get_graph().print_ascii()
print(search_agent.get_graph().draw_mermaid())