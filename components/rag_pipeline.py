from langchain.chains import RetrievalQA

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode,tools_condition
from components.agent import State,agent,retriever_tool,rewrite,generate,grade_documents
from langchain_core.messages import HumanMessage



def rag_query(query: str,test=0):
    graph_builder = StateGraph(State)

    graph_builder.add_node("agent", agent)
    retriever = ToolNode([retriever_tool])
    graph_builder.add_node("retriever", retriever)
    graph_builder.add_node("rewrite", rewrite)

    graph_builder.add_node("generate", generate)

    graph_builder.add_edge(START, "agent")

    graph_builder.add_conditional_edges(
        "agent",

        tools_condition,
        {
            "tools": "retriever",
            END: END
        }
    )

    graph_builder.add_conditional_edges(
        "retriever",
        grade_documents
    )

    graph_builder.add_edge("generate", END)
    graph_builder.add_edge("rewrite", "agent")

    graph = graph_builder.compile()
    query = {"messages": [HumanMessage(query)]}

    from pprint import pprint
    for output in graph.stream(query):
        for key, value in output.items():
            if test==1:
                pprint(f"Output from node '{key}':")
                pprint(value["messages"][-1])
            final_state = value
        if test == 1:
            pprint("------")

    messages = final_state["messages"]
    final_message = messages[-1]
    return final_message