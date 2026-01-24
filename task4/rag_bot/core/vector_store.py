# core/vector_store.py
"""
Модуль для работы с векторной базой данных ChromaDB
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Изменено!
from langchain_core.documents import Document  # Изменено!

from config import CONFIG, DEFAULT_SEARCH_K, DEFAULT_SCORE_THRESHOLD

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Менеджер векторного хранилища"""

    def __init__(self, config=None):
        """
        Инициализация менеджера векторного хранилища

        Args:
            config: Конфигурация векторного хранилища
        """
        self.config = config or CONFIG.vector_store
        self._client = None
        self._embedding_function = None
        self._vector_store = None

        logger.info(f"Инициализация VectorStoreManager: {self.config.persist_directory}")

    def _get_embedding_function(self) -> HuggingFaceEmbeddings:
        """Создание функции для генерации эмбеддингов"""
        if self._embedding_function is None:
            try:
                logger.info(f"Загрузка модели эмбеддингов: {self.config.embedding_model}")

                self._embedding_function = HuggingFaceEmbeddings(
                    model_name=self.config.embedding_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={
                        'normalize_embeddings': True
                    },
                    cache_folder=str(CONFIG.llm.model_cache_dir)
                )

                logger.info("Модель эмбеддингов загружена успешно")

            except Exception as e:
                logger.error(f"Ошибка загрузки модели эмбеддингов: {e}")
                raise

        return self._embedding_function

    def get_vector_store(self) -> Chroma:
        """
        Получение экземпляра векторного хранилища

        Returns:
            Chroma: Экземпляр векторного хранилища
        """
        if self._vector_store is None:
            try:
                embedding_function = self._get_embedding_function()

                # Проверяем существование директории
                if not self.config.persist_directory.exists():
                    logger.warning(f"Директория с векторной базой не найдена: {self.config.persist_directory}")
                    logger.info("Будет создана новая коллекция при первом добавлении документов")

                logger.info(f"Подключение к коллекции: {self.config.collection_name}")

                self._vector_store = Chroma(
                    persist_directory=str(self.config.persist_directory),
                    collection_name=self.config.collection_name,
                    embedding_function=embedding_function,
                    client_settings=Settings(
                        anonymized_telemetry=False,
                        is_persistent=True,
                        chroma_server_host='localhost',
                        chroma_server_http_port=8000
                    )
                )

                logger.info("Векторное хранилище успешно инициализировано")

            except Exception as e:
                logger.error(f"Ошибка инициализации векторного хранилища: {e}")
                raise

        return self._vector_store

    # core/vector_store.py - исправленный метод similarity_search
    def similarity_search(
            self,
            query: str,
            k: Optional[int] = None,
            score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        vector_store = self.get_vector_store()

        # Используем значения из конфигурации по умолчанию
        search_k = k or self.config.search_kwargs.get("k", DEFAULT_SEARCH_K)
        threshold = score_threshold or self.config.search_kwargs.get(
            "score_threshold", DEFAULT_SCORE_THRESHOLD
        )

        # Убедимся, что threshold не None
        if threshold is None:
            threshold = DEFAULT_SCORE_THRESHOLD

        logger.info(f"Поиск по запросу: '{query[:50]}...', k={search_k}, threshold={threshold}")

        try:
            # Выполняем поиск
            docs_with_scores = vector_store.similarity_search_with_score(
                query=query,
                k=search_k
            )

            # Фильтрация и форматирование результатов
            results = []
            for doc, score in docs_with_scores:
                # Безопасное сравнение с обработкой None
                score_val = float(score) if score is not None else 0.0
                threshold_val = float(threshold) if threshold is not None else 0.0

                if score_val >= threshold_val:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score_val
                    })
                    logger.debug(f"Найден документ: score={score_val:.3f}, source={doc.metadata.get('source', 'N/A')}")

            logger.info(f"Найдено документов: {len(results)} (всего получено: {len(docs_with_scores)})")

            # Сортируем по score (от большего к меньшему)
            results.sort(key=lambda x: x['score'], reverse=True)

            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Получение информации о коллекции

        Returns:
            Dict: Информация о коллекции
        """
        try:
            vector_store = self.get_vector_store()
            collection = vector_store._collection

            count = collection.count()

            # Пытаемся получить метаданные коллекции
            metadata = {}
            try:
                # Используем приватный атрибут
                metadata = collection._client.get_collection(
                    name=collection.name
                ).metadata or {}
            except:
                pass

            return {
                "name": collection.name,
                "count": count,
                "metadata": metadata,
                "status": "available" if count > 0 else "empty"
            }

        except Exception as e:
            logger.error(f"Ошибка получения информации о коллекции: {e}")
            return {
                "name": self.config.collection_name,
                "count": 0,
                "metadata": {},
                "status": "error",
                "error": str(e)
            }

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Добавление документов в векторную базу

        Args:
            documents: Список документов для добавления

        Returns:
            List[str]: ID добавленных документов
        """
        try:
            vector_store = self.get_vector_store()
            ids = vector_store.add_documents(documents)
            logger.info(f"Добавлено {len(ids)} документов")
            return ids
        except Exception as e:
            logger.error(f"Ошибка добавления документов: {e}")
            return []

    def delete_collection(self) -> bool:
        """
        Удаление коллекции

        Returns:
            bool: Успешно ли удаление
        """
        try:
            vector_store = self.get_vector_store()
            vector_store._client.delete_collection(vector_store._collection.name)
            self._vector_store = None  # Сбрасываем кэш
            logger.info(f"Коллекция {self.config.collection_name} удалена")
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления коллекции: {e}")
            return False