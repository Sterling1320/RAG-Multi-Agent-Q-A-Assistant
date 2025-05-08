# Optional CLI (for testing without UI)
from agents import Agent

if __name__ == "__main__":
    agent = Agent()
    while True:
        query = input("Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = agent.run(query)
        print("Answer:", response["answer"])
        print("Tool used:", response["used_tool"])
        print("Context:", response["context"])
