from components.retriever import get_vectorstore
from typing import Annotated, Sequence, TypedDict, Literal
from pydantic import BaseModel, Field


from langchain.tools.retriever import create_retriever_tool
from langgraph.graph.message import add_messages
from langchain import hub
from langchain_core.messages import  BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434")

retriever = get_vectorstore().as_retriever(search_type="similarity", search_kwargs = {'k': 5})

#To use it as an agent, you must make the retriever a tool.
retriever_tool=create_retriever_tool(retriever, "kendo", "returns information about the kendo introductions")
tools=[retriever_tool]


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def grade_documents(state) -> Literal["generate", "rewrite"]:

    # Data model
    class grade(BaseModel):

        binary_score: str = Field(description="similarity result")

    # preconfiguration of output
    llm_with_structured_output = llm.with_structured_output(grade)

    # The prompt is important here; it should explain the role of the model and what it should do very well.
    prompt = PromptTemplate(
        template="""
        You are an AI assistant tasked with grading the relevance of a document in response to a user question.

        ## Context:
        - A document has been retrieved from a knowledge base.
        - A user has asked a question.

        ## Objective:
        - Assess whether the content of the document is relevant to the user's question.
        - Consider both direct keyword overlap and semantic similarity (i.e., similar meaning even if different words are used).

        ## Instructions:
        - If the document contains information that answers, supports, or is clearly related to the user's question, return: 'yes'
        - If the document does NOT provide relevant or helpful information, return: 'no'

        ## Document:
        {context}

        ## User Question:
        {question}

        ## Your Answer (only respond with 'yes' or 'no'):
        """,
        input_variables=["context", "question"]
    )

    # Chain
    chain = prompt | llm_with_structured_output

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score
    print(score)
    if score == "yes":
        return "generate"

    else:
        return "rewrite"

def agent(state):

    messages = state["messages"]

    llm_with_tools = llm.bind_tools(tools, tool_choice="required")
    response = llm_with_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    messages = state["messages"]
    question = messages[0].content

    rewrite_prompt = PromptTemplate.from_template("""
    You are an intelligent assistant helping to improve user questions for better information retrieval.

    ## Objective:
    Analyze the user's original question and reformulate it to be:
    - Clear and unambiguous
    - Focused on the key intent
    - Optimized for retrieving relevant documents

    ## Original Question:
    {question}

    ## Rewritten Question:
    (Provide only the improved version of the question without explanations)
    """)

    prompt_text = rewrite_prompt.format(question=question)

    msg = [HumanMessage(content=prompt_text)]

    response = llm.invoke(msg)

    return {"messages": [response]}




def generate(state):

    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt from langchain lib.
    prompt = hub.pull("rlm/rag-prompt")

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

