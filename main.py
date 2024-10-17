from query import query
def start():
    ask()

def ask():
    while True:
        user_input = input("Q:")
        response= query(user_input)
        print("A:", response)

if __name__=="__main__":
    start()