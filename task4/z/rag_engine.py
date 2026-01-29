import sys
import logging
from pathlib import Path

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Импорты LangChain и библиотек ---
try:
    from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
    from langchain_core.example_selectors import LengthBasedExampleSelector
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    # ИСПРАВЛЕНИЕ: Добавлен AutoModelForCausalLM
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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
        "answer": "Step 1: Analyze the question - looking for economic activity in Crystalbrook.\nStep 2: Review documents - found information about herbal remedies and alchemical supplies.\nStep 3: Extract and synthesize information - Crystalbrook's main economic activity is herbal remedies and alchemical supplies.\nStep 4: Provide answer with citation [1].\n\nA. Herbal remedies and alchemical supplies are Crystalbrook's main economic activities."
    },
    {
        "question": "Who leads the defenders of Crystalbrook?",
        "answer": "Step 1: Analyze the question - looking for the leader of Crystalbrook's defenders.\nStep 2: Review documents - found information about Taraix.\nStep 3: Extract and synthesize information - Taraix leads the defenders of Crystalbrook.\nStep 4: Provide answer with citation [2].\n\nA. Taraix leads the defenders of Crystalbrook."
    },
    {
        "question": "What limits moonleaf harvesting in DECR_002?",
        "answer": "Step 1: Analyze the question - looking for limitations on moonleaf harvesting.\nStep 2: Review documents - found information about designated areas and specific lunar phases.\nStep 3: Extract and synthesize information - moonleaf harvesting is limited to designated areas during specific lunar phases.\nStep 4: Provide answer with citation [3].\n\nA. Moonleaf harvesting in DECR_002 is limited to designated areas during specific lunar phases."
    },
    {
        "question": "What happens if you violate DECR_001?",
        "answer": "Step 1: Analyze the question - looking for consequences of violating DECR_001.\nStep 2: Review documents - found information about community service requirements.\nStep 3: Extract and synthesize information - violating DECR_001 results in community service requirements.\nStep 4: Provide answer with citation [4].\n\nA. If you violate DECR_001, you are subject to community service requirements."
    },
    {
        "question": "What does the Crystal Essence primarily do?",
        "answer": "Step 1: Analyze the question - looking for the primary function of Crystal Essence.\nStep 2: Review documents - found information about accelerating reflexes and perception.\nStep 3: Extract and synthesize information - Crystal Essence primarily accelerates reflexes and perception.\nStep 4: Provide answer with citation [5].\n\nA. The Crystal Essence primarily accelerates reflexes and perception."
    },
    {
        "question": "Who prepares the Crystal Essence exclusively?",
        "answer": "Step 1: Analyze the question - looking for who prepares Crystal Essence exclusively.\nStep 2: Review documents - found information about Lorekeeper Syloos.\nStep 3: Extract and synthesize information - Lorekeeper Syloos prepares the Crystal Essence exclusively.\nStep 4: Provide answer with citation [6].\n\nA. Lorekeeper Syloos prepares the Crystal Essence exclusively."
    },
    {
        "question": "What happened to Lugous near the border?",
        "answer": "Step 1: Analyze the question - looking for events involving Lugous near the border.\nStep 2: Review documents - found information about a skirmish with Imperial scouts.\nStep 3: Extract and synthesize information - Lugous had a skirmish with Imperial scouts near the border.\nStep 4: Provide answer with citation [7].\n\nA. Lugous had a skirmish with Imperial scouts near the border."
    },
    {
        "question": "Why are Lugous's dreams more vivid?",
        "answer": "Step 1: Analyze the question - looking for reason for vivid dreams of Lugous.\nStep 2: Review documents - found information about visiting the crystal cave.\nStep 3: Extract and synthesize information - Lugous's dreams are more vivid since visiting the crystal cave.\nStep 4: Provide answer with citation [8].\n\nA. Lugous's dreams are more vivid since visiting the crystal cave."
    },
    {
        "question": "What is the Legend of the Twin Moons about?",
        "answer": "Step 1: Analyze the question - looking for information about the Legend of the Twin Moons.\nStep 2: Review documents - found information about the gift of the twin moons.\nStep 3: Extract and synthesize information - The Legend of the Twin Moons is about the gift of the twin moons.\nStep 4: Provide answer with citation [9].\n\nA. The Legend of the Twin Moons is about the gift of the twin moons."
    },
    {
        "question": "Who is the hero in the MYTH_001 legend?",
        "answer": "Step 1: Analyze the question - looking for the hero in MYTH_001 legend.\nStep 2: Review documents - found information about the Stone Speaker.\nStep 3: Extract and synthesize information - The hero in the MYTH_001 legend is the Stone Speaker.\nStep 4: Provide answer with citation [10].\n\nA. The hero in the MYTH_001 legend is the Stone Speaker."
    },
    {
        "question": "What does Sunstone Elixir enhance?",
        "answer": "Step 1: Analyze the question - looking for what Sunstone Elixir enhances.\nStep 2: Review documents - found information about temporary vitality and endurance.\nStep 3: Extract and synthesize information - Sunstone Elixir enhances temporary vitality and endurance.\nStep 4: Provide answer with citation [11].\n\nA. Sunstone Elixir enhances temporary vitality and endurance."
    },
    {
        "question": "What requires quarrymaster approval in DECR_003?",
        "answer": "Step 1: Analyze the question - looking for what requires quarrymaster approval in DECR_003.\nStep 2: Review documents - found information about using standing stones for construction.\nStep 3: Extract and synthesize information - Using standing stones for construction requires quarrymaster approval in DECR_003.\nStep 4: Provide answer with citation [12].\n\nA. Using standing stones for construction requires quarrymaster approval in DECR_003."
    },
    {
        "question": "Who does Taraix write to in LETT_003?",
        "answer": "Step 1: Analyze the question - looking for who Taraix writes to in LETT_003.\nStep 2: Review documents - found information about a trusted friend.\nStep 3: Extract and synthesize information - Taraix writes to a trusted friend in LETT_003.\nStep 4: Provide answer with citation [13].\n\nA. Taraix writes to a trusted friend in LETT_003."
    },
    {
        "question": "What is REPO_002's security classification?",
        "answer": "Step 1: Analyze the question - looking for REPO_002's security classification.\nStep 2: Review documents - found information about RESTRICTED ACCESS.\nStep 3: Extract and synthesize information - REPO_002's security classification is RESTRICTED ACCESS.\nStep 4: Provide answer with citation [14].\n\nA. REPO_002's security classification is RESTRICTED ACCESS."
    },
    {
        "question": "What is the primary trend in REPO_006?",
        "answer": "Step 1: Analyze the question - looking for the primary trend in REPO_006.\nStep 2: Review documents - found information about Imperial forces consolidating positions.\nStep 3: Extract and synthesize information - The primary trend in REPO_006 is that Imperial forces are consolidating positions.\nStep 4: Provide answer with citation [15].\n\nA. The primary trend in REPO_006 is that Imperial forces are consolidating positions."
    },
    {
        "question": "What does the Elderwood Extract enhance?",
        "answer": "Step 1: Analyze the question - looking for what Elderwood Extract enhances.\nStep 2: Review documents - found information about cognitive processing and memory.\nStep 3: Extract and synthesize information - The Elderwood Extract enhances cognitive processing and memory.\nStep 4: Provide answer with citation [16].\n\nA. The Elderwood Extract enhances cognitive processing and memory."
    },
    {
        "question": "What does Lugous discover while exploring?",
        "answer": "Step 1: Analyze the question - looking for what Lugous discovers while exploring.\nStep 2: Review documents - found information about a hidden cave.\nStep 3: Extract and synthesize information - Lugous discovers a hidden cave while exploring.\nStep 4: Provide answer with citation [17].\n\nA. Lugous discovers a hidden cave while exploring."
    },
    {
        "question": "What does Syloos show in journal entries?",
        "answer": "Step 1: Analyze the question - looking for what Syloos shows in journal entries.\nStep 2: Review documents - found information about a new herbal preparation.\nStep 3: Extract and synthesize information - Syloos shows a new herbal preparation in journal entries.\nStep 4: Provide answer with citation [18].\n\nA. Syloos shows a new herbal preparation in journal entries."
    },
    {
        "question": "What is the moral of MYTH_002?",
        "answer": "Step 1: Analyze the question - looking for the moral of MYTH_002.\nStep 2: Review documents - found information about greatest treasures being overlooked.\nStep 3: Extract and synthesize information - The moral of MYTH_002 is that the greatest treasures are often overlooked.\nStep 4: Provide answer with citation [19].\n\nA. The moral of MYTH_002 is that the greatest treasures are often overlooked."
    },
    {
        "question": "What must be preserved per DECR_004?",
        "answer": "Step 1: Analyze the question - looking for what must be preserved per DECR_004.\nStep 2: Review documents - found information about historical sites over development.\nStep 3: Extract and synthesize information - Historical sites must be preserved over development per DECR_004.\nStep 4: Provide answer with citation [20].\n\nA. Historical sites must be preserved over development per DECR_004."
    },
    {
        "question": "What is the capital of France?",
        "answer": "Step 1: Analyze the question - looking for the capital of France.\nStep 2: Review documents - no relevant information found in provided documents.\nStep 3: Since no relevant information is available, respond with 'No confirmations found'.\n\nA. No confirmations found."
    }
]


# system_instruction должен быть просто текстом инструкции.
system_instruction = """
### Role
You are a large English‑language LLM assistant.  
Your task is to carefully answer the user's question using **ONLY** the information from the provided list of documents.  
If the documents do not contain the necessary information, honestly state "No confirmations found".  
Avoid speculation and hallucinations.

### Workflow steps
1. Read all documents from the `<Documents>` block carefully.  
2. Identify which of them are truly relevant to the question.  
3. Summarize the key facts (you may take notes for yourself, but do not show them to the user).  
4. Formulate the final answer in English, relying only on verified facts.  
5. At the end of the answer, add citation markers in the form `[1]`, `[2]` — these are the document numbers from the `<Documents>` block that confirm a specific statement.

### Chain-of-Thought (CoT) reasoning
When answering questions, follow these steps:
Step 1: Analyze the question and identify what information is being requested.
Step 2: Review the provided documents to find relevant information.
Step 3: If relevant information is found, extract and synthesize it to answer the question.
Step 4: If no relevant information is found in the documents, respond with "No confirmations found".

### Information Quality Check
Before providing an answer, evaluate the relevance and quality of retrieved documents:
- If documents are completely irrelevant to the question, respond with "No confirmations found"
- If documents contain partial information but not enough to fully answer the question, respond with "No confirmations found"
- Only provide answers when documents contain sufficient and relevant information

### Output format
The answer must be complete but concise:  
**A. Brief answer** (1–3 sentences).  

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
        temperature=0.1,
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
    # создаём template для примеров
    example_template = """User question: {question}
    Answer: {answer}
    """
    # создаём промпт из шаблона выше
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template=example_template)
    
    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=250  # параметром выставляется максимальная длина примера
    )

    # а suffix - это вопрос пользователя и поле для ответа
    suffix = """
    ### `<Documents>`
    {context}
    … (up to N documents possible)

    ### `<User question>`
    {user_question}

    ### `<Your answer>`
    (Follow the A. format as described above)"""

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,  # используем example_selector вместо examples
        example_prompt=example_prompt,
        prefix=system_instruction,
        suffix=suffix,
        input_variables=["user_question", "context"],
        example_separator="\n" # символ, которым будем разделять примеры (перенос строки)
    )
    # filled_examples = few_shot_prompt.format()

    # 6. Сборка Цепочки
    def format_docs(docs):
        # Добавляем проверку на пустые документы
        if not docs:
            return ""
        return "\n\n".join([d.page_content for d in docs])
    
    # Добавляем проверку релевантности документов перед генерацией
    def check_relevance_and_generate(input_dict):
        context = input_dict["context"]
        user_question = input_dict["user_question"]
        
        # Если нет контекста, сразу возвращаем "No confirmations found"
        if not context or len(context.strip()) == 0:
            return "No confirmations found"
        
        # Если контекст есть, продолжаем с обычной цепочкой
        return few_shot_prompt.invoke(input_dict)
    
    rag_chain = (
            {
                "context": retriever | format_docs,
                "user_question": RunnablePassthrough()
            }
            | check_relevance_and_generate
            | llm
            | StrOutputParser()
    )

    logging.info("RAG Chain assembled successfully.")
    return rag_chain