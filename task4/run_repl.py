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
            logging.info("Вопрос не может быть пустым. Попробуйте снова.")
            continue
        if query.lower() == "exit":
            logging.info("Выход из программы.")
            break

        response = engine.search(query)
        results = response["results"]
        answer = response["answer"]

        logging.debug("\n=== Результаты ===")
        for idx, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "неизвестно")
            filename = doc.metadata.get("filename", "неизвестно")
            logging.debug(f"\n#{idx}")
            logging.debug(f"Источник: {source}")
            logging.debug(f"Файл: {filename}")
            logging.debug(doc.page_content)

        logging.info("\n=== Ответ LLM ===")
        logging.info(answer)


if __name__ == "__main__":
    run_repl()
