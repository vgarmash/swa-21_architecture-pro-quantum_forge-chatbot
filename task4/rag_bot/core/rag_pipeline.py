"""
Основной пайплайн RAG
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import logging

from .vector_store import VectorStoreManager
from .llm_client import QwenLLMClient
from .prompt_templates import PromptBuilder
from config import CONFIG, DEFAULT_SCORE_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Ответ RAG пайплайна"""
    answer: str
    retrieved_docs: list
    processing_time: float
    used_sources: list
    has_context: bool


class RAGPipeline:
    """Основной пайплайн RAG"""

    def __init__(self, config=None):
        """
        Инициализация RAG пайплайна

        Args:
            config: Конфигурация RAG
        """
        self.config = config or CONFIG
        self.vector_store = VectorStoreManager(self.config.vector_store)
        self.llm_client = QwenLLMClient(self.config.llm)
        self.prompt_builder = PromptBuilder(self.config)

        # Инициализация компонентов
        self._initialize_components()

    def _initialize_components(self):
        """Инициализация всех компонентов пайплайна"""
        print("Инициализация RAG пайплайна...")

        # Проверяем векторное хранилище
        try:
            info = self.vector_store.get_collection_info()
            print(f"Векторное хранилище: {info['name']} ({info['count']} документов)")
        except Exception as e:
            print(f"Внимание: Ошибка доступа к векторному хранилищу: {e}")

        # Загружаем модель (может занять время)
        print("Загрузка языковой модели...")
        self.llm_client.load_model()

        print("RAG пайплайн готов к работе")

    def process_query(
            self,
            query: str,
            k: int = 5,
            similarity_threshold: Optional[float] = None
    ) -> RAGResponse:
        """
        Обработка запроса через RAG пайплайн

        Args:
            query: Вопрос пользователя
            k: Количество документов для поиска
            similarity_threshold: Порог схожести для фильтрации

        Returns:
            RAGResponse: Ответ пайплайна
        """
        start_time = time.time()

        try:
            # Устанавливаем значение по умолчанию для порога если не задано
            if similarity_threshold is None:
                similarity_threshold = DEFAULT_SCORE_THRESHOLD

            # Шаг 1: Поиск релевантных документов
            retrieved_docs = self.vector_store.similarity_search(query, k=k)

            # Шаг 2: Фильтрация по порогу схожести с безопасной проверкой
            filtered_docs = []
            for doc in retrieved_docs:
                score = doc.get('score')
                # Безопасное сравнение
                if (score is not None and
                        isinstance(score, (int, float)) and
                        score >= similarity_threshold):
                    filtered_docs.append(doc)

            # Шаг 3: Определяем, есть ли контекст для ответа
            has_context = len(filtered_docs) > 0

            # Шаг 4: Строим промпт и генерируем ответ
            if has_context:
                prompt = self.prompt_builder.build_rag_prompt(query, filtered_docs)
            else:
                prompt = self.prompt_builder.build_no_context_prompt(query)

            # Шаг 5: Генерация ответа
            answer = self.llm_client.generate_response(prompt)

            # Шаг 6: Извлекаем использованные источники
            used_sources = []
            if has_context:
                used_sources = list(set(
                    doc.get('metadata', {}).get('source', 'Неизвестно')
                    for doc in filtered_docs
                ))

            processing_time = time.time() - start_time

            return RAGResponse(
                answer=answer,
                retrieved_docs=filtered_docs,
                processing_time=processing_time,
                used_sources=used_sources,
                has_context=has_context
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
            return RAGResponse(
                answer=f"Произошла ошибка при обработке запроса: {str(e)}",
                retrieved_docs=[],
                processing_time=processing_time,
                used_sources=[],
                has_context=False
            )

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Получение информации о пайплайне

        Returns:
            Dict: Информация о компонентах пайплайна
        """
        try:
            vector_store_info = self.vector_store.get_collection_info()
        except Exception:
            vector_store_info = {"error": "Не доступно"}

        return {
            "vector_store": vector_store_info,
            "llm_model": self.config.llm.model_name,
            "embedding_model": self.config.vector_store.embedding_model,
            "few_shot_examples": len(self.config.few_shot.get_selected_examples())
        }
import signal
import functools

class TimeoutError(Exception):
    """Исключение для таймаута"""
    pass

def timeout(seconds):
    """Декоратор для ограничения времени выполнения"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Функция {func.__name__} превысила время выполнения ({seconds} сек)")

            # Устанавливаем таймер
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                # Отключаем таймер
                signal.alarm(0)

            return result
        return wrapper
    return decorator