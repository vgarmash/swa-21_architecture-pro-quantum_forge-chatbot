import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Импорты LangChain и библиотек ---
try:
    from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    # ИСПРАВЛЕНИЕ: Добавлен AutoModelForCausalLM
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
    import torch
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    sys.exit(1)

# --- КОНФИГУРАЦИЯ ---

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# Модели
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat" # Или Qwen/Qwen2.5-1.5B-Instruct

# --- ПРИМЕРЫ ДЛЯ FEW-SHOT ---
# Обратите внимание: формат ответа должен соответствовать тому, чего вы хотите от модели.
# Если нужно CoT (Step 1...), добавьте его в answer.

examples = [
    {
        "query": "What are main ingredients and effects of Crystal Blade?",
        "context": "The primary ingredient is Crystal way harvested during the twin suns' alignment. Effects include temporary enhanced vitality, accelerated movement, and temporary resilience.",
        "answer": "The primary ingredient is Crystal way harvested during the twin suns' alignment. Effects include temporary enhanced vitality, accelerated movement, and temporary resilience."
    },
    {
        "query": "Why is Morac permanently strong without drinking Crystal Blade?",
        "context": "Chronic exposure (as in the case of Morac) leads to permanent strength but requires frequent nourishment. source_docs: [\"MYTH_001\", \"ENCY_001\"]",
        "answer": "Morac fell into a cauldron of Crystal Blade as a baby, giving him permanent superhuman strength."
    }
]

# ИСПРАВЛЕНИЕ: Убрали плейсхолдеры {context} и {question} отсюда.
# system_instruction должен быть просто текстом инструкции.
system_instruction = """
You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise.
"""

def get_rag_chain():
    """
    Инициализирует модели, загружает ChromaDB и собирает RAG пайплайн.
    """
    logging.info("Initializing RAG Engine...")

    # 1. Инициализация устройства
    device = 0 if torch.cuda.is_available() else -1
    logging.info(f"Device: {'CUDA' if device == 0 else 'CPU'}")

    # 2. Эмбеддинги
    logging.info(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    # 3. LLM (Phi-3 / CausalLM)
    logging.info(f"Loading Generative Model: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)

    # Настройка стоп-токенов
    eos_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.3,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_token_id,
        device=device,
        # <--- ИСПРАВЛЕНИЕ: Отключаем кэш, чтобы ушла ошибка DynamicCache
        model_kwargs={"use_cache": False}
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # 4. Загрузка ChromaDB
    logging.info(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")
    if not CHROMA_DB_PATH.exists():
        logging.error(f"Database folder not found: {CHROMA_DB_PATH}")
        raise FileNotFoundError(f"Database not found at {CHROMA_DB_PATH}. Please run build_index.py first.")

    db = Chroma(
        persist_directory=str(CHROMA_DB_PATH),
        embedding_function=embeddings,
        collection_name="knowledge_base", # Убедитесь, что build_index.py использует такое же имя!
        collection_metadata={"hnsw:space": "cosine"}
    )

    # --- ПРОВЕРКА ДАННЫХ ---
    try:
        count = db._collection.count()
        logging.info(f"*** DATABASE CHECK: Found {count} embeddings in the collection. ***")
        if count == 0:
            logging.warning("!!! WARNING: The database appears to be empty! !!!")
    except Exception as e:
        logging.error(f"Could not check DB count: {e}")

    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 5. Сборка Промптов
    example_prompt = PromptTemplate(
        input_variables=["query", "context", "answer"],
        template="Context: {context}\nQ: {query}\nA: {answer}\n"
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="", suffix="", example_separator="\n"
    )
    filled_examples = few_shot_prompt.format()

    # ИСПРАВЛЕНИЕ: Здесь мы явно формируем промпт, вставляя {context} и {query} в нужные места.
    # Теперь переменные context и query подставятся корректно.
    full_prompt = PromptTemplate(
        template="""{instruction}

Examples:
{examples}

Context: {context}
Question: {query}
Answer:""",
        input_variables=["instruction", "examples", "context", "query"]
    )

    # 6. Сборка Цепочки
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
            {
                "context": retriever | format_docs,
                "query": RunnablePassthrough(),
                "instruction": lambda x: system_instruction,
                "examples": lambda x: filled_examples
            }
            | full_prompt
            | llm
            | StrOutputParser()
    )

    logging.info("RAG Chain assembled successfully.")
    return rag_chain