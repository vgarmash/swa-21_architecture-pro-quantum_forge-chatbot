from rag_engine import get_rag_chain

def main():
    try:
        chain = get_rag_chain()
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    print("\n" + "="*50)
    print("RAG REPL Interface (Type 'exit' to quit)")
    print("="*50 + "\n")

    while True:
        query = input("User: ").strip()
        if query.lower() in ['exit', 'quit']:
            break
        if not query: continue

        # Запуск цепочки
        try:
            print("Bot: ", end="")
            result = chain.invoke(query)
            print(result)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()