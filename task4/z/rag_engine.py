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
    from langchain_core.prompts.example_selector import LengthBasedExampleSelector
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

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# Модели
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- ПРИМЕРЫ ДЛЯ FEW-SHOT ---
# Обратите внимание: формат ответа должен соответствовать тому, чего вы хотите от модели.
# Если нужно CoT (Step 1...), добавьте его в answer.

examples = [
    {
        "question": "What is Crystalbrook's main economic activity?",
        "answer": "Herbal remedies and alchemical supplies."
    },
    {
        "question": "Who leads the defenders of Crystalbrook?",
        "answer": "Taraix."
    },
    {
        "question": "What limits moonleaf harvesting in DECR_002?",
        "answer": "Only in designated areas during specific lunar phases."
    },
    {
        "question": "What happens if you violate DECR_001?",
        "answer": "Subject to community service requirements."
    },
    {
        "question": "What does the Crystal Essence primarily do?",
        "answer": "Accelerates reflexes and perception."
    },
    {
        "question": "Who prepares the Crystal Essence exclusively?",
        "answer": "Lorekeeper Syloos."
    },
    {
        "question": "What happened to Lugous near the border?",
        "answer": "A skirmish with Imperial scouts."
    },
    {
        "question": "Why are Lugous's dreams more vivid?",
        "answer": "Since visiting the crystal cave."
    },
    {
        "question": "What is the Legend of the Twin Moons about?",
        "answer": "The gift of the twin moons."
    },
    {
        "question": "Who is the hero in the MYTH_001 legend?",
        "answer": "The Stone Speaker."
    },
    {
        "question": "What does Sunstone Elixir enhance?",
        "answer": "Temporary vitality and endurance."
    },
    {
        "question": "What requires quarrymaster approval in DECR_003?",
        "answer": "Using standing stones for construction."
    },
    {
        "question": "Who does Taraix write to in LETT_003?",
        "answer": "A trusted friend."
    },
    {
        "question": "What is REPO_002's security classification?",
        "answer": "RESTRICTED ACCESS."
    },
    {
        "question": "What is the primary trend in REPO_006?",
        "answer": "Imperial forces are consolidating positions."
    },
    {
        "question": "What does the Elderwood Extract enhance?",
        "answer": "Cognitive processing and memory."
    },
    {
        "question": "What does Lugous discover while exploring?",
        "answer": "A hidden cave."
    },
    {
        "question": "What does Syloos show in journal entries?",
        "answer": "A new herbal preparation."
    },
    {
        "question": "What is the moral of MYTH_002?",
        "answer": "The greatest treasures are often overlooked."
    },
    {
        "question": "What must be preserved per DECR_004?",
        "answer": "Historical sites over development."
    }
]

# ИСПРАВЛЕНИЕ: Убрали плейсхолдеры {context} и {question} отсюда.
# system_instruction должен быть просто текстом инструкции.
system_instruction = """
### Role
You are a large English‑language LLM assistant.  
Your task is to carefully answer the user’s question using **ONLY** the information from the provided list of documents.  
If the documents do not contain the necessary information, honestly state “No confirmations found”.  
Avoid speculation and hallucinations.

### Workflow steps
1. Read all documents from the `<Documents>` block carefully.  
2. Identify which of them are truly relevant to the question.  
3. Summarize the key facts (you may take notes for yourself, but do not show them to the user).  
4. Formulate the final answer in English, relying only on verified facts.  
5. At the end of the answer, add citation markers in the form `[1]`, `[2]` — these are the document numbers from the `<Documents>` block that confirm a specific statement.

### Output format
The answer must consist of two parts:  
**A. Brief answer** (1–3 sentences).  
**B. Detailed explanation** (in bullet points), where each statement is accompanied by a source citation in square brackets.

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

    # 3. LLM
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

### `<Documents>`
`[1]` {{doc1_title}} — {{doc1_excerpt}}  
`[2]` {{doc2_title}} — {{doc2_excerpt}}  
`[3]` {{doc3_title}} — {{doc3_excerpt}}  
… (up to N documents possible)

### `<User question>`
{{user_question}}

### `<Your answer>`
(Follow the A. and B. format as described above)""",
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