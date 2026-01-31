import hashlib
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

try:  # pragma: no cover - prefer dedicated package
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal installs
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:  # pragma: no cover - prefer native wrapper
    from langchain_chroma import Chroma
except ModuleNotFoundError:  # pragma: no cover - fallback for older installs
    from langchain_community.vectorstores import Chroma

# Ensure access to task3 modules when running from task6
CURRENT_DIR = Path(__file__).resolve().parent
TASK3_DIR = CURRENT_DIR.parent / "task3"
if str(TASK3_DIR) not in sys.path:
    sys.path.insert(0, str(TASK3_DIR))

from config import (  # type: ignore  # noqa: E402
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_PATH,
)
from build_index import load_documents, prepare_documents_for_index  # type: ignore  # noqa: E402

LOG_PATH = CURRENT_DIR / "update_index.log"

def configure_logging() -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def compute_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_local_index() -> dict[str, dict[str, object]]:
    local_index: dict[str, dict[str, object]] = {}
    for file_path in sorted(KNOWLEDGE_BASE_PATH.glob("*.txt")):
        local_index[file_path.name] = {
            "path": file_path,
            "hash": compute_hash(file_path),
        }
    return local_index


def build_remote_index(vector_store: Chroma) -> dict[str, dict[str, object]]:
    remote_index: dict[str, dict[str, object]] = {}
    payload = vector_store.get(include=["metadatas"]) or {}
    ids = payload.get("ids") or []
    metadatas = payload.get("metadatas") or []

    for doc_id, metadata in zip(ids, metadatas):
        if not metadata:
            continue
        filename = metadata.get("filename")
        if not filename:
            continue
        entry = remote_index.setdefault(
            filename,
            {"ids": [], "doc_hash": metadata.get("doc_hash"), "metadatas": []},
        )
        entry["ids"].append(doc_id)
        entry["metadatas"].append(metadata)
        if metadata.get("doc_hash"):
            entry["doc_hash"] = metadata["doc_hash"]
    return remote_index


def update_index() -> None:
    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Папка {KNOWLEDGE_BASE_PATH} не найдена. Добавьте документы перед обновлением индекса."
        )

    configure_logging()

    start_time = time.perf_counter()
    logging.info("Index update started | knowledge base=%s", KNOWLEDGE_BASE_PATH)

    local_index = build_local_index()
    local_filenames = set(local_index.keys())

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_PATH.resolve()),
        embedding_function=embeddings,
    )

    remote_index = build_remote_index(vector_store)
    remote_filenames = set(remote_index.keys())

    missing_files = sorted(remote_filenames - local_filenames)
    new_or_changed_files: list[str] = []
    unchanged_files: list[str] = []

    for filename in sorted(local_filenames):
        local_hash = local_index[filename]["hash"]
        remote_hash = remote_index.get(filename, {}).get("doc_hash")
        if remote_hash == local_hash:
            unchanged_files.append(filename)
        else:
            new_or_changed_files.append(filename)

    removed_chunks = 0
    if missing_files:
        logging.info("Removing entries for missing files: %s", ", ".join(missing_files))
        for filename in missing_files:
            ids_to_remove = remote_index.get(filename, {}).get("ids") or []
            if ids_to_remove:
                vector_store.delete(ids=list(ids_to_remove))
                removed_chunks += len(ids_to_remove)

    for filename in new_or_changed_files:
        existing_ids = remote_index.get(filename, {}).get("ids") or []
        if existing_ids:
            vector_store.delete(ids=list(existing_ids))
            removed_chunks += len(existing_ids)

    documents_to_process: list = []
    if new_or_changed_files:
        all_documents = load_documents(KNOWLEDGE_BASE_PATH)
        docs_by_filename = {doc.metadata.get("filename"): doc for doc in all_documents}
        for filename in new_or_changed_files:
            document = docs_by_filename.get(filename)
            if not document:
                logging.warning("Файл %s не удалось загрузить, пропуск", filename)
                continue
            document.metadata["doc_hash"] = local_index[filename]["hash"]
            document.metadata["source"] = str(local_index[filename]["path"])
            documents_to_process.append(document)

    added_chunks = 0
    if documents_to_process:
        chunk_documents, _stats = prepare_documents_for_index(documents_to_process)
        if chunk_documents:
            ids_to_add: list[str] = []
            section_counters: defaultdict[str, int] = defaultdict(int)
            for doc in chunk_documents:
                metadata = doc.metadata
                filename = metadata.get("filename", "unknown")
                section_index = metadata.get("section_index")
                if section_index is None:
                    section_index = section_counters[filename]
                chunk_index = metadata.get("chunk_index")
                if chunk_index is None:
                    chunk_index = section_counters[filename]
                ids_to_add.append(
                    f"{filename}::sec{int(section_index):03d}::chunk{int(chunk_index):03d}"
                )
                section_counters[filename] += 1
            vector_store.add_documents(documents=chunk_documents, ids=ids_to_add)
            added_chunks = len(chunk_documents)

    if hasattr(vector_store, "persist"):
        vector_store.persist()
    elif hasattr(vector_store, "_client") and hasattr(vector_store._client, "persist"):
        vector_store._client.persist()

    total_chunks = -1
    if hasattr(vector_store, "_collection") and hasattr(vector_store._collection, "count"):
        total_chunks = vector_store._collection.count()
    elif hasattr(vector_store, "_client"):
        client = vector_store._client  # type: ignore[attr-defined]
        if hasattr(client, "_collection") and hasattr(client._collection, "count"):
            total_chunks = client._collection.count()  # type: ignore[attr-defined]

    elapsed = time.perf_counter() - start_time
    logging.info(
        "Index update completed | changed=%d | removed_files=%d | unchanged=%d | added_chunks=%d | removed_chunks=%d | total_chunks=%d | elapsed=%.2fs",
        len(new_or_changed_files),
        len(missing_files),
        len(unchanged_files),
        added_chunks,
        removed_chunks,
        total_chunks,
        elapsed,
    )


if __name__ == "__main__":
    try:
        update_index()
    except Exception:  # pragma: no cover - fail safe logging
        logging.exception("Index update failed")
        raise
