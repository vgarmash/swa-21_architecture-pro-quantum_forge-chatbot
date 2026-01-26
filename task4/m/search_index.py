import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT_TEMPLATE = (
    "You are a helpful assistant. Provide accurate answer using only the context below. "
    "If the answer is not in the context, say that the information is insufficient.\n\n"
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


def search() -> None:
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
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    logging.info("Загрузка LLM: %s", LLM_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    while True:
        query = input("Введите вопрос (или 'exit' для выхода): ").strip()
        if not query:
            print("Вопрос не может быть пустым. Попробуйте снова.")
            continue
        if query.lower() == "exit":
            print("Выход из программы.")
            break

        logging.info("Поиск релевантных чанков (k=2).")
        results = vector_store.similarity_search(query, k=2)

        print("\n=== Результаты ===")
        for idx, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "неизвестно")
            filename = doc.metadata.get("filename", "неизвестно")
            print(f"\n#{idx}")
            print(f"Источник: {source}")
            print(f"Файл: {filename}")
            print(doc.page_content)

        context = "\n\n".join(
            f"[{idx}] {doc.page_content}" for idx, doc in enumerate(results, start=1)
        )
        prompt = PROMPT_TEMPLATE.format(question=query, context=context)

        logging.info("Генерация ответа с помощью LLM.")
        outputs = generator(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0,
        )
        generated_text = outputs[0]["generated_text"]
        answer = generated_text[len(prompt) :].strip() if generated_text.startswith(prompt) else generated_text.strip()

        print("\n=== Ответ LLM ===")
        print(answer)


if __name__ == "__main__":
    search()
