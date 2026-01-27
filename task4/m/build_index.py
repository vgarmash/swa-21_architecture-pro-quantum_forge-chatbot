import logging
import os
import time
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PROJECT_ROOT = Path(__file__).parent.parent.parent
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def load_documents(directory: Path) -> list[Document]:
    documents: list[Document] = []
    for file_path in sorted(directory.glob("*.txt")):
        loader = TextLoader(str(file_path), encoding="utf-8")
        file_docs = loader.load()
        if not file_docs:
            continue
        doc = file_docs[0]
        doc.metadata.update(
            {
                "source": str(file_path),
                "filename": file_path.name,
            }
        )
        documents.append(doc)
    return documents


def build_index() -> None:
    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Папка {KNOWLEDGE_BASE_PATH} не найдена. Создайте ее и добавьте чанки."
        )

    start_time = time.perf_counter()
    logging.info("Старт индексации. Папка: %s", KNOWLEDGE_BASE_PATH)

    documents = load_documents(KNOWLEDGE_BASE_PATH)
    if not documents:
        raise ValueError("В папке knowledge_base не найдено .txt файлов.")

    total_chars = sum(len(doc.page_content) for doc in documents)
    logging.info(
        "Загружено документов: %s. Суммарный размер текста: %s символов.",
        len(documents),
        total_chars,
    )
    logging.info("Инициализация эмбеддингов: %s", EMBEDDING_MODEL)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logging.info(
        "Создание индекса Chroma. Коллекция: %s, директория: %s",
        COLLECTION_NAME,
        CHROMA_DB_PATH,
    )
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(Path(CHROMA_DB_PATH).resolve()),
        collection_metadata={
            "hnsw:space": "cosine",
            "model": EMBEDDING_MODEL,
            "embedding_dim": str(EMBEDDING_DIMENSIONS)
        }
    )


    elapsed = time.perf_counter() - start_time
    logging.info(
        "Индексация завершена. Добавлено чанков: %s. Время: %.2f сек.",
        len(documents),
        elapsed,
    )


if __name__ == "__main__":
    build_index()
