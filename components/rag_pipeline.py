from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from typing import TypedDict
from components.retriever import get_vectorstore
from components.search_tool import web_search
from langchain_core.prompts import ChatPromptTemplate




llm = ChatOllama(
    model="mistral:7b-instruct",
    base_url="http://localhost:11434"
)
vector_store= get_vectorstore()
retreiver = vector_store.as_retriever(search_type="similarity", k=5)

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are an intelligent multilingual assistant. Respond in the same language as the user's question.

User Question:
{question}

Context:
{context}

Based on the context, answer accurately and clearly in the user's language. If the context is insufficient, say so.
""")


# State
class AgentState(TypedDict):
    input: str
    llm_output: str
    search_output: str
    context: str
    grader_decision: str


# === Nodes ===
def rag_node(state: AgentState) -> AgentState:
    print(">> RAG Node çalışıyor...")
    docs = retreiver.invoke(state["input"])
    context = "\n\n".join([doc.page_content for doc in docs])
    state["context"] = context
    return state

def should_fallback(state: AgentState) -> str:
    if state["context"].strip() == "":
        return "web_search"
    else:
        return "llm_rag"

def grader_node(state: AgentState) -> AgentState:
    print(">> Grader Node is working...")
    question = state["input"]
    context = state["context"]

    grader_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant that grades the relevance of a document to a user question.

## Context
- Document: {context}
- Question: {question}

## Task:
Respond ONLY with one word, either:
- yes → if the document is relevant
- no → if the document is not relevant

Respond with ONLY one word. Do NOT include any bullet points, explanations, or formatting.
""")

    formatted = grader_prompt.invoke({"question": question, "context": context})
    response = llm.invoke(formatted)
    decision = response.content.strip().lower()

    state["grader_decision"] = decision
    return state

def route_after_grader(state: AgentState) -> str:

    if state.get("grader_decision") == "yes":
        return "llm_rag"
    else:
        return "web_search"

def llm_rag_node(state: AgentState) -> AgentState:
    question = state["input"]
    context = state["context"]
    response = llm.invoke(prompt.invoke({"question": question, "context": context}))
    state["llm_output"] = response.content
    return state

def llm_node(state: AgentState) -> AgentState:
    question = state["input"]
    context = state.get("search_output", "")

    full_prompt = prompt.invoke({
        "question": question,
        "context": context
    })

    llm_response = llm.invoke(full_prompt)
    state["llm_output"] = llm_response.content
    return state

def search_node(state: AgentState) -> AgentState:
    question = state["input"]
    context = web_search(question)
    state["search_output"] = context
    return state


# === Build Graph ===
graph = StateGraph(AgentState)
graph.add_node("rag", rag_node)
graph.add_node("llm_rag", llm_rag_node)
graph.add_node("grader", grader_node)
graph.add_node("web_search", search_node)
graph.add_node("llm", llm_node)


graph.set_entry_point("rag")
graph.add_edge("rag", "grader")
graph.add_conditional_edges("grader", route_after_grader, {
    "llm_rag": "llm_rag",
    "web_search": "web_search"
})
graph.add_edge("web_search", "llm")
graph.set_finish_point("llm_rag")
graph.set_finish_point("llm")


search_agent = graph.compile()