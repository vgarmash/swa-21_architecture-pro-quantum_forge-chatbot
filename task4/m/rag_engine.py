import logging
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
CHROMA_DIR = Path(CHROMA_DB_PATH)
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_MODEL = "Qwen/Qwen1.5-1.8B-Chat"

PROMPT_INSTRUCTIONS = (
    "You are a helpful assistant. Answer in English using only the context below. "
    "Do not quote or copy the context verbatim; paraphrase and synthesize it. "
    "Do not include the context, metadata, file paths, hashes, or headings in the answer. "
    "If the answer is not in the context, say that the information is insufficient and do not make up an answer. "
    "If you are uncertain or don't know the answer, respond with: 'Information is unavailable. Ask another question.'\n\n"
    "Output format: 1-3 sentences followed by citations like [1], [2] referring to the context document numbers. (see format examples below) "
    "Every factual statement must have at least one citation. Return only the answer, nothing else.\n\n"
    "Here are few examples (style only; do not copy sources or filenames):\n"
)

FEW_SHOT_EXAMPLES_RAW = [
    {
        "question": "What is Crystalbrook known for?",
        "answer": "Its humid climate and remarkable resistance to Imperial occupation.\n\n[1] ENCY_005.txt",
    },
    {
        "question": "Who leads the defenders of Crystalbrook?",
        "answer": "Taraix leads the 21 trained combatants.\n\n[1] ENCY_005.txt",
    },
    {
        "question": "What are the primary effects of Starlight Tonic?",
        "answer": "Accelerated reflexes and perception.\n\n[1] ENCY_010.txt",
    },
    {
        "question": "What is the main ingredient of Starlight Tonic?",
        "answer": "Crystalized dew from dream-fed plants.\n\n[1] ENCY_010.txt",
    },
    {
        "question": "What did Lugous and Imperial scouts have?",
        "answer": "A skirmish near the border.\n\n[1] JOUR_001.txt",
    },
    {
        "question": "How have Lugous's dreams changed?",
        "answer": "They became more vivid after visiting a crystal cave.\n\n[1] JOUR_001.txt",
    },
    {
        "question": "What does the Council of Elders limit the harvesting of?",
        "answer": "Moonleaf to designated areas during specific lunar phases.\n\n[1] DECR_002.txt",
    },
    {
        "question": "What is the primary ingredient of Elderwood Extract?",
        "answer": "Whispering willow bark from the Elder Grove.\n\n[1] ENCY_006.txt",
    },
    {
        "question": "What effects does Elderwood Extract provide?",
        "answer": "Enhanced cognitive processing and memory.\n\n[1] ENCY_006.txt",
    },
    {
        "question": "Who broke a tool according to Quenix's journal?",
        "answer": "Quenix himself broke another tool.\n\n[1] JOUR_010.txt",
    },
    {
        "question": "What did Taraix discover according to myth?",
        "answer": "A cave that whispered secrets (in MYTH_003).\n\n[1] MYTH_003.txt",
    },
    {
        "question": "What legend explains why mountains hold memory?",
        "answer": "The legend of The Dream Weaver seeking answers.\n\n[1] MYTH_002.txt",
    },
    {
        "question": "Who requires approval to use standing stones for construction?",
        "answer": "Quarrymaster approval is required.\n\n[1] DECR_003.txt",
    },
    {
        "question": "What does Taraix employ strategically against Imperial forces?",
        "answer": "He employs Crystal Essence (Sunstone Elixir).\n\n[1] ENCY_015.txt",
    },
    {
        "question": "What does Lugous ask to be burned in his letter?",
        "answer": "He asks the recipient to burn the letter after reading.\n\n[1] LETT_001.txt",
    },
    {
        "question": "What is the main trend in the Cultural Analysis Report?",
        "answer": "Resource extraction exceeds sustainable levels.\n\n[1] REPO_001.txt",
    },
    {
        "question": "What is the key observation in the Economic Assessment Report?",
        "answer": "Increased Imperial activity along northern borders.\n\n[1] REPO_003.txt",
    },
    {
        "question": "What does the Strategic Evaluation Report indicate?",
        "answer": "Depletion of certain natural resources.\n\n[1] REPO_008.txt",
    },
    {
        "question": "What must public gatherings exceeding twenty individuals do?",
        "answer": "They require advance notification.\n\n[1] DECR_001.txt",
    },
    {
        "question": "Who is the legendary figure in The Gift of the Twin Moons?",
        "answer": "The Stone Speaker (in MYTH_001) or Syloos (in MYTH_005).\n\n[1] MYTH_001.txt\n[2] MYTH_005.txt",
    },
]


def _sanitize_few_shot_examples(examples: list[dict]) -> list[dict]:
    pattern = re.compile(r"\[(\d+)\]\s+[A-Za-z0-9_\-]+\.txt")
    sanitized = []
    for example in examples:
        answer = pattern.sub(r"[\1]", example["answer"])
        sanitized.append({"question": example["question"], "answer": answer})
    return sanitized


FEW_SHOT_EXAMPLES = _sanitize_few_shot_examples(FEW_SHOT_EXAMPLES_RAW)

EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\nAnswer: {answer}\n",
)
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0


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
            collection_metadata={"hnsw:space": "cosine"}
        )

        logging.info("Загрузка LLM: %s", LLM_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
        self._generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            return_full_text=False,
        )
        self._llm = HuggingFacePipeline(pipeline=self._generator)

    def search(self, query: str, k: int = 2) -> dict:
        if not query:
            raise ValueError("Question cannot be empty.")

        logging.info("Поиск релевантных чанков (k=%s).", k)
        results = self._vector_store.similarity_search(query, k=k)

        # Проверка, есть ли релевантные результаты
        if not results:
            return {
                "query": query,
                "results": [],
                "context": "",
                "answer": "Nothing found. Ask another question.",
            }

        def format_docs(docs):
            return "\n\n".join(
                f"[{idx}] {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
                for idx, doc in enumerate(docs, start=1)
            )

        context = format_docs(results)
        prompt = FewShotPromptTemplate(
            examples=FEW_SHOT_EXAMPLES,
            example_prompt=EXAMPLE_PROMPT,
            example_separator="\n",
            prefix=PROMPT_INSTRUCTIONS,
            suffix=(
                "\nQuestion: {question}\n\n"
                "Context:\n{context}\n\n"
                "Answer:"
            ),
            input_variables=["question", "context"],
        )
        retriever = self._vector_store.as_retriever(search_kwargs={"k": k})

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self._llm
            | StrOutputParser()
        )

        logging.info("Генерация ответа с помощью RAG-цепочки.")
        answer = rag_chain.invoke(query).strip()

        # Дополнительная проверка: если ответ содержит признаки вымышленной информации
        # (например, слишком общие фразы или отсутствие конкретики)
        if self._is_fabricated_answer(answer, query, results):
            answer = "Can't answer properly. Ask another question."

        return {
            "query": query,
            "results": results,
            "context": context,
            "answer": answer,
        }
    
    def _is_fabricated_answer(self, answer: str, query: str, results: list) -> bool:
        """Checks whether the answer is fabricated."""
        if not answer or len(answer.strip()) < 10:
            return True

        answer_lower = answer.lower()
        if "information is unavailable" in answer_lower or "ask another question" in answer_lower:
            return False

        cited_indices = self._extract_citations(answer)
        if not cited_indices:
            return True

        max_index = len(results)
        if any(idx < 1 or idx > max_index for idx in cited_indices):
            return True

        # Ослабляем проверку пересечения токенов: достаточно валидных цитат.
        return False

    def _extract_citations(self, text: str) -> set:
        matches = []
        current = ""
        inside = False
        for ch in text:
            if ch == "[":
                inside = True
                current = ""
                continue
            if ch == "]" and inside:
                inside = False
                if current.isdigit():
                    matches.append(int(current))
                continue
            if inside:
                current += ch
        return set(matches)

    def _extract_keywords(self, text: str) -> set:
        stopwords = {
            "the", "and", "or", "to", "of", "a", "an", "in", "on", "for", "with",
            "is", "are", "was", "were", "be", "been", "by", "at", "as", "it", "this",
            "that", "from", "but", "not", "into", "than", "then", "these", "those",
            "information", "unavailable", "ask", "another", "question"
        }
        tokens = []
        current = []
        for ch in text.lower():
            if ch.isalnum():
                current.append(ch)
            elif current:
                token = "".join(current)
                tokens.append(token)
                current = []
        if current:
            tokens.append("".join(current))

        return {t for t in tokens if len(t) > 2 and t not in stopwords}
