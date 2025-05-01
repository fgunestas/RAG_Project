from components.rag_pipeline import rag_query

if __name__ == "__main__":
    user_query = input("Soru: ")
    response = rag_query(user_query,test=1)
    print("\nYanıt:\n", response)