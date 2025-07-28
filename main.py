from components.rag_pipeline import search_agent

if __name__ == "__main__":
    while True:
        user_input = input("Write Prompt: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        result = search_agent.invoke({"input": user_input})
        print("\n=== Response ===")
        print(result['final_output'])
