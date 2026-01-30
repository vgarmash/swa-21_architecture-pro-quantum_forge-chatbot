import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Настройки OpenAI API
OPENAI_BASE_URL = "http://localhost:8040/api/v1"
OPENAI_API_KEY = "lemonade"
OPENAI_MODEL = "Mistral-7B-v0.3-Instruct-Hybrid"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def create_llm_pipeline():
    logging.info("Инициализация OpenAI LLM: %s", OPENAI_MODEL)
    """Create and return a ChatOpenAI instance for text generation."""
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=OPENAI_MODEL,
        temperature=0.0,
    )
    return llm