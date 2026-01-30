import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from chonkie import RecursiveChunker  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    RecursiveChunker = None  # type: ignore

from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_PATH,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    TARGET_CHUNK_TOKENS,
    CHUNK_OVERLAP_RATIO,
)

CHUNK_OVERLAP_TOKENS = int(TARGET_CHUNK_TOKENS * CHUNK_OVERLAP_RATIO)

HEADING_REGEX = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")
ALT_HEADING_REGEX = re.compile(
    r"^(?P<title>(?:Chapter|Section|Глава|Раздел)\b[^\n]*)$", re.IGNORECASE
)
SENTENCE_BOUNDARY_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-Я0-9\"'\[])")
BULLET_REGEX = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class Section:
    title: str
    level: int
    content: str
    index: int


def load_documents(directory: Path) -> list[Document]:
    documents: list[Document] = []
    for file_path in sorted(directory.glob("*.txt")):
        loader = TextLoader(str(file_path), encoding="utf-8")
        file_docs = loader.load()
        if not file_docs:
            continue
        doc = file_docs[0]
        doc.page_content = doc.page_content.strip()
        doc.metadata.update(
            {
                "source": str(file_path),
                "filename": file_path.name,
            }
        )
        documents.append(doc)
    return documents


def estimate_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\s\w]", text))


def split_into_sentences(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    sentences = [segment.strip() for segment in SENTENCE_BOUNDARY_REGEX.split(cleaned) if segment.strip()]
    if not sentences:
        return [cleaned]
    return sentences


def extract_sections(content: str) -> list[Section]:
    sections: list[Section] = []
    current_lines: list[str] = []
    current_title = "Document"
    current_level = 1
    section_index = 0

    lines = content.splitlines()
    for line in lines:
        heading_match = HEADING_REGEX.match(line.strip())
        alt_match = ALT_HEADING_REGEX.match(line.strip()) if not heading_match else None

        if heading_match or alt_match:
            if current_lines:
                sections.append(
                    Section(
                        title=current_title,
                        level=current_level,
                        content="\n".join(current_lines).strip(),
                        index=section_index,
                    )
                )
                section_index += 1
                current_lines = []

            if heading_match:
                current_title = heading_match.group("title").strip()
                current_level = len(heading_match.group("hashes"))
            else:
                current_title = alt_match.group("title").strip() if alt_match else current_title
                current_level = 2
            continue

        current_lines.append(line)

    if current_lines:
        sections.append(
            Section(
                title=current_title,
                level=current_level,
                content="\n".join(current_lines).strip(),
                index=section_index,
            )
        )

    if not sections:
        sections.append(Section(title="Document", level=1, content=content.strip(), index=0))

    return sections


def analyze_document(document: Document, sections: Iterable[Section]) -> None:
    token_count = estimate_token_count(document.page_content)
    bullet_count = len(BULLET_REGEX.findall(document.page_content))
    section_list = list(sections)
    sample_titles = ", ".join(section.title for section in section_list[:5]) or "n/a"
    logging.info(
        "Document analysis | file=%s | approx_tokens=%d | sections=%d | bullet_lists=%d | sample_sections=%s",
        document.metadata.get("filename"),
        token_count,
        len(section_list),
        bullet_count,
        sample_titles,
    )


def create_recursive_chunker() -> Optional[object]:
    if RecursiveChunker is None:
        logging.info("chonkie RecursiveChunker not available. Using fallback sentence-aware chunker.")
        return None

    chunk_overlap = int(TARGET_CHUNK_TOKENS * CHUNK_OVERLAP_RATIO)

    if hasattr(RecursiveChunker, "from_recipe"):
        try:
            chunker = RecursiveChunker.from_recipe(
                "markdown",
                lang="en",
                chunk_size=TARGET_CHUNK_TOKENS,
                overlap=chunk_overlap,
                tokenizer="tiktoken",
            )
            logging.info("Using chonkie RecursiveChunker markdown recipe.")
            return chunker
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning(
                "RecursiveChunker.from_recipe failed (%s). Attempting direct initialization.",
                exc,
            )

    try:
        chunker = RecursiveChunker(
            chunk_size=TARGET_CHUNK_TOKENS,
            chunk_overlap=chunk_overlap,
        )
        logging.info("Using chonkie RecursiveChunker with direct configuration.")
        return chunker
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Failed to initialize RecursiveChunker (%s). Falling back to internal chunker.", exc)
        return None


def chunk_with_recursive_chunker(chunker: Optional[object], text: str) -> Optional[list[str]]:
    if chunker is None:
        return None
    try:
        raw_chunks = chunker.chunk(text)
    except AttributeError:
        try:
            raw_chunks = chunker.split_text(text)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("RecursiveChunker split_text failed (%s). Fallback activated.", exc)
            return None
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("RecursiveChunker chunk failed (%s). Fallback activated.", exc)
        return None

    if not raw_chunks:
        return []

    if isinstance(raw_chunks, dict):
        raw_chunks = [raw_chunks.get("text", "")]
    if isinstance(raw_chunks, str):
        raw_chunks = [raw_chunks]

    processed: list[str] = []
    for chunk in raw_chunks:
        if isinstance(chunk, dict):
            chunk_text = str(chunk.get("text", "")).strip()
        else:
            chunk_text = str(chunk).strip()
        if chunk_text:
            processed.append(chunk_text)
    return processed


def fallback_chunk_section(text: str) -> list[tuple[str, int]]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[tuple[str, int]] = []
    i = 0
    while i < len(sentences):
        chunk_sentences: list[str] = []
        chunk_tokens = 0
        start_i = i

        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = estimate_token_count(sentence)
            if chunk_tokens and chunk_tokens + sentence_tokens > MAX_CHUNK_TOKENS and chunk_tokens >= MIN_CHUNK_TOKENS:
                break

            if not chunk_tokens and sentence_tokens > MAX_CHUNK_TOKENS:
                chunk_sentences.append(sentence)
                chunk_tokens = sentence_tokens
                i += 1
                break

            chunk_sentences.append(sentence)
            chunk_tokens += sentence_tokens
            i += 1

            if chunk_tokens >= TARGET_CHUNK_TOKENS:
                break

        chunk_text = " ".join(chunk_sentences).strip()
        chunk_tokens = estimate_token_count(chunk_text)
        chunks.append((chunk_text, chunk_tokens))

        if i >= len(sentences):
            break

        overlap_tokens = 0
        overlap_sentences = 0
        for sentence in reversed(chunk_sentences):
            overlap_tokens += estimate_token_count(sentence)
            overlap_sentences += 1
            if overlap_tokens >= CHUNK_OVERLAP_TOKENS:
                break
        i = max(start_i + len(chunk_sentences) - overlap_sentences, start_i + 1)

    return chunks


def merge_small_chunks(chunks: list[str], token_counts: list[int]) -> tuple[list[str], list[int]]:
    if not chunks:
        return chunks, token_counts

    merged_chunks: list[str] = []
    merged_counts: list[int] = []

    for text, count in zip(chunks, token_counts):
        if merged_chunks and count < MIN_CHUNK_TOKENS:
            merged_chunks[-1] = f"{merged_chunks[-1].strip()}\n\n{text.strip()}".strip()
            merged_counts[-1] = estimate_token_count(merged_chunks[-1])
        else:
            merged_chunks.append(text)
            merged_counts.append(count)

    return merged_chunks, merged_counts


def chunk_section_text(text: str, chunker: Optional[object]) -> tuple[list[str], list[int]]:
    section_text = text.strip()
    if not section_text:
        return [], []

    chunk_texts = chunk_with_recursive_chunker(chunker, section_text)
    if chunk_texts:
        token_counts = [estimate_token_count(chunk) for chunk in chunk_texts]
        return merge_small_chunks(chunk_texts, token_counts)

    fallback_chunks = fallback_chunk_section(section_text)
    chunk_texts = [chunk for chunk, _ in fallback_chunks if chunk]
    token_counts = [tokens for _, tokens in fallback_chunks if tokens > 0]
    return merge_small_chunks(chunk_texts, token_counts)


def prepare_documents_for_index(documents: list[Document]) -> tuple[list[Document], dict[str, float]]:
    chunker = create_recursive_chunker()
    chunk_documents: list[Document] = []
    token_counts: list[int] = []
    total_sections = 0

    for doc in documents:
        sections = extract_sections(doc.page_content)
        analyze_document(doc, sections)
        total_sections += len(sections)

        for section in sections:
            section_chunks, section_token_counts = chunk_section_text(section.content, chunker)
            for chunk_index, (chunk_text, token_count) in enumerate(zip(section_chunks, section_token_counts)):
                if not chunk_text:
                    continue
                token_counts.append(token_count)
                metadata = dict(doc.metadata)
                metadata.update(
                    {
                        "section_title": section.title,
                        "section_level": section.level,
                        "section_index": section.index,
                        "chunk_index": chunk_index,
                        "approx_token_count": token_count,
                    }
                )
                chunk_documents.append(Document(page_content=chunk_text, metadata=metadata))

    stats: dict[str, float] = {
        "documents": len(documents),
        "sections": total_sections,
        "chunks": len(chunk_documents),
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "avg_tokens": (sum(token_counts) / len(token_counts)) if token_counts else 0.0,
    }

    return chunk_documents, stats


def reset_vector_store(path: Path) -> None:
    if path.exists():
        logging.info("Resetting Chroma directory at %s", path)
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_index() -> None:
    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Папка {KNOWLEDGE_BASE_PATH} не найдена. Создайте ее и добавьте чанки."
        )

    start_time = time.perf_counter()
    logging.info("Index build started | knowledge base directory=%s", KNOWLEDGE_BASE_PATH)

    documents = load_documents(KNOWLEDGE_BASE_PATH)
    if not documents:
        raise ValueError("В папке knowledge_base не найдено .txt файлов.")

    total_chars = sum(len(doc.page_content) for doc in documents)
    logging.info(
        "Documents loaded | total=%d | total_characters=%d",
        len(documents),
        total_chars,
    )

    chunk_documents, stats = prepare_documents_for_index(documents)
    if not chunk_documents:
        raise ValueError("No chunks were generated from the knowledge base documents.")

    logging.info(
        "Chunking summary | sections=%d | chunks=%d | avg_tokens=%.1f | min_tokens=%d | max_tokens=%d",
        stats["sections"],
        stats["chunks"],
        stats["avg_tokens"],
        stats["min_tokens"],
        stats["max_tokens"],
    )

    logging.info("Resetting vector store at %s", CHROMA_DB_PATH)
    reset_vector_store(CHROMA_DB_PATH)

    logging.info("Initializing embeddings: %s", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logging.info(
        "Creating Chroma index | collection=%s",
        COLLECTION_NAME,
    )
    Chroma.from_documents(
        documents=chunk_documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(Path(CHROMA_DB_PATH).resolve()),
        collection_metadata={
            "hnsw:space": "cosine",
            "model": EMBEDDING_MODEL,
            "embedding_dim": str(EMBEDDING_DIMENSIONS),
        },
    )

    elapsed = time.perf_counter() - start_time
    logging.info(
        "Index build completed | chunks=%d | elapsed=%.2fs",
        stats["chunks"],
        elapsed,
    )


if __name__ == "__main__":
    build_index()
