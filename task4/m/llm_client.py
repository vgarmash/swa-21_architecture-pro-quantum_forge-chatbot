import logging
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

LLM_MODEL = "Qwen/Qwen1.5-1.8B-Chat"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def create_llm_pipeline():
    logging.info("Загрузка LLM: %s", LLM_MODEL)
    """Create and return a HuggingFacePipeline for text generation."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=False,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=generator)