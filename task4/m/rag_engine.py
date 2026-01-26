import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_MODEL = "Qwen/Qwen1.5-1.8B-Chat"

PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the question in clear, natural, human language "
    "based only on the context below. Do not quote or copy the context verbatim; "
    "paraphrase and synthesize it. If the answer is not in the context, say that the "
    "information is insufficient.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:"
)
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class RagEngine:
    def __init__(self) -> None:
        logging.info("Загрузка chromadb")
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Папка {CHROMA_DIR} не найдена. Сначала запустите build_index.py."
            )

        logging.info("Инициализация эмбеддингов: %s", EMBEDDING_MODEL)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        logging.info(
            "Подключение к Chroma. Коллекция: %s, директория: %s",
            COLLECTION_NAME,
            CHROMA_DIR,
        )
        self._vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )

        logging.info("Загрузка LLM: %s", LLM_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
        self._generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def search(self, query: str, k: int = 2) -> dict:
        if not query:
            raise ValueError("Вопрос не может быть пустым.")

        logging.info("Поиск релевантных чанков (k=%s).", k)
        results = self._vector_store.similarity_search(query, k=k)

        context = "\n\n".join(
            f"[{idx}] {doc.page_content}" for idx, doc in enumerate(results, start=1)
        )
        prompt = PROMPT_TEMPLATE.format(question=query, context=context)

        logging.info("Генерация ответа с помощью LLM.")
        outputs = self._generator(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0,
        )
        generated_text = outputs[0]["generated_text"]
        answer = (
            generated_text[len(prompt) :].strip()
            if generated_text.startswith(prompt)
            else generated_text.strip()
        )

        return {
            "query": query,
            "results": results,
            "context": context,
            "answer": answer,
        }
