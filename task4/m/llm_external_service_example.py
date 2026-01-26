from typing import Optional

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

BASE_URL = "https://litellm.devzone.su/v1"
LITELLLM_API_KEY = "sk-1xrqYLR_1JJHDvmASbz1eQ"
DEFAULT_MODEL = "openrouter/anthropic/claude-3.5-sonnet"


def get_api_key() -> str:
    api_key = LITELLLM_API_KEY
    if not api_key:
        raise RuntimeError(
            "Переменная окружения LITELLM_API_KEY не задана. "
            "Установите ключ доступа к внешнему сервису."
        )
    return api_key


def call_via_openai(prompt: str, model: Optional[str] = None) -> str:
    client = OpenAI(
        api_key=get_api_key(),
        base_url=BASE_URL,
    )
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def call_via_langchain(prompt: str, model: Optional[str] = None) -> str:
    llm = ChatOpenAI(
        api_key=get_api_key(),
        base_url=BASE_URL,
        model=model or DEFAULT_MODEL,
        temperature=0.2,
    )
    result = llm.invoke([HumanMessage(content=prompt)])
    return getattr(result, "content", "") or ""


if __name__ == "__main__":
    test_prompt = "Скажи привет одной строкой."

    print("--- OpenAI SDK ---")
    print(call_via_openai(test_prompt))

    print("--- LangChain ---")
    print(call_via_langchain(test_prompt))
