from pathlib import Path

import chromadb


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / "chroma_db"

    client = chromadb.PersistentClient(path=str(db_path))
    collections = client.list_collections()
    if not collections:
        print("Векторные коллекции не найдены.")
        return

    collection = collections[0]
    print(f"Используется коллекция: {collection.name}")

    while True:
        query = input("Введите запрос (/exit для выхода): ").strip()
        if query == "/exit":
            print("Завершение работы.")
            break

        result = collection.query(query_texts=[query], n_results=5)
        documents = result.get("documents", [[]])
        if not documents or not documents[0]:
            print("Нет результатов.")
            continue

        print("Результаты:")
        for index, document in enumerate(documents[0], start=1):
            print(f"{index}. {document}")


if __name__ == "__main__":
    main()
