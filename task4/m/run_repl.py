import logging

from rag_engine import RagEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def run_repl() -> None:
    engine = RagEngine()

    while True:
        query = input("Введите вопрос (или 'exit' для выхода): ").strip()
        if not query:
            print("Вопрос не может быть пустым. Попробуйте снова.")
            continue
        if query.lower() == "exit":
            print("Выход из программы.")
            break

        response = engine.search(query)
        results = response["results"]
        answer = response["answer"]

        print("\n=== Результаты ===")
        for idx, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "неизвестно")
            filename = doc.metadata.get("filename", "неизвестно")
            print(f"\n#{idx}")
            print(f"Источник: {source}")
            print(f"Файл: {filename}")
            print(doc.page_content)

        print("\n=== Ответ LLM ===")
        print(answer)


if __name__ == "__main__":
    run_repl()
