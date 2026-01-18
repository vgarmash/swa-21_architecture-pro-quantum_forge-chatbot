"""
build_index.py - Создание векторного индекса базы знаний
Версия для Linux с Python 3.12 и LangChain 1.x
"""

import os
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    UnstructuredFileLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

class KnowledgeBaseIndexer:
    """
    Класс для создания векторного индекса базы знаний
    Использует модель all-MiniLM-L6-v2 и ChromaDB
    """

    def __init__(self,
                 knowledge_base_path: str = "knowledge_base",
                 persist_directory: str = "chroma_db",
                 chunk_size: int = 800,
                 chunk_overlap: int = 150):
        """
        Инициализация индексатора

        Args:
            knowledge_base_path: Путь к папке с документами
            persist_directory: Папка для сохранения индекса
            chunk_size: Размер чанка в символах
            chunk_overlap: Перекрытие чанков в символах
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.persist_directory = Path(persist_directory)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Инициализация модели эмбеддингов
        # Модель: all-MiniLM-L6-v2
        # Репозиторий: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        # Размер эмбеддингов: 384
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Можно изменить на 'cuda' если есть GPU
            encode_kwargs={
                'normalize_embeddings': True,
                'show_progress_bar': True
            }
        )

        # Инициализация текстового сплиттера
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            keep_separator=True
        )

        # Статистика
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_documents': 0,
            'total_chunks': 0,
            'model_info': {
                'name': 'sentence-transformers/all-MiniLM-L6-v2',
                'embedding_dimensions': 384,
                'url': 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2'
            }
        }

    def validate_environment(self) -> bool:
        """Проверка наличия необходимых папок и файлов"""
        print("🔍 Проверка окружения...")

        if not self.knowledge_base_path.exists():
            print(f"❌ Ошибка: Папка '{self.knowledge_base_path}' не найдена!")
            print(f"   Создайте папку и поместите туда документы")
            return False

        # Создаем папку для индекса
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        print(f"✓ Папка с документами: {self.knowledge_base_path}")
        print(f"✓ Папка для индекса: {self.persist_directory}")
        print(f"✓ Модель эмбеддингов: {self.stats['model_info']['name']}")
        print(f"✓ Размерность эмбеддингов: {self.stats['model_info']['embedding_dimensions']}")

        return True

    def load_documents(self) -> List[Document]:
        """Загрузка документов из различных форматов"""
        print("\n📂 Загрузка документов...")

        documents = []
        supported_extensions = {
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.pdf': PyPDFLoader,
        }

        # Подсчет файлов по типам
        file_counts = {ext: 0 for ext in supported_extensions}

        for ext, loader_class in supported_extensions.items():
            try:
                # Ищем файлы с данным расширением
                pattern = f"**/*{ext}"
                loader = DirectoryLoader(
                    str(self.knowledge_base_path),
                    glob=pattern,
                    loader_cls=loader_class,
                    show_progress=True,
                    use_multithreading=True,
                    silent_errors=True
                )

                loaded_docs = loader.load()
                file_counts[ext] = len(loaded_docs)
                documents.extend(loaded_docs)

                if loaded_docs:
                    print(f"  ✓ {ext}: {len(loaded_docs)} файлов")

            except Exception as e:
                print(f"  ⚠️  Ошибка загрузки {ext} файлов: {e}")

        # Загрузка других файлов через UnstructuredFileLoader
        try:
            other_files = list(self.knowledge_base_path.rglob("*"))
            other_files = [f for f in other_files if f.is_file() and
                           f.suffix.lower() not in supported_extensions and
                           f.suffix.lower() not in ['.py', '.json', '.yaml', '.yml']]

            if other_files:
                print(f"  📄 Загрузка других форматов файлов...")
                for file_path in other_files:
                    try:
                        loader = UnstructuredFileLoader(str(file_path))
                        loaded_docs = loader.load()
                        documents.extend(loaded_docs)
                        print(f"    ✓ {file_path.name}")
                    except Exception as e:
                        print(f"    ⚠️  Ошибка загрузки {file_path.name}: {e}")

        except Exception as e:
            print(f"  ⚠️  Ошибка при загрузке других файлов: {e}")

        self.stats['total_documents'] = len(documents)

        print(f"\n📊 Статистика загрузки:")
        for ext, count in file_counts.items():
            if count > 0:
                print(f"  {ext}: {count} файлов")
        print(f"  Всего документов: {len(documents)}")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Разделение документов на чанки с сохранением метаданных"""
        print(f"\n✂️  Разделение документов на чанки...")
        print(f"   Размер чанка: {self.chunk_size} символов")
        print(f"   Перекрытие: {self.chunk_overlap} символов")

        all_chunks = []

        for doc_idx, doc in enumerate(documents):
            # Разделяем документ на чанки
            doc_chunks = self.text_splitter.split_documents([doc])

            # Обогащаем метаданные каждого чанка
            for chunk_idx, chunk in enumerate(doc_chunks):
                # Получаем исходные метаданные
                source_path = chunk.metadata.get('source', 'unknown')
                source_file = Path(source_path).name

                # Генерируем уникальный ID чанка
                chunk_id = hashlib.md5(
                    f"{source_file}_{doc_idx}_{chunk_idx}".encode()
                ).hexdigest()[:12]

                # Обновляем метаданные
                chunk.metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_idx,
                    'total_chunks_in_doc': len(doc_chunks),
                    'document_index': doc_idx,
                    'source_file': source_file,
                    'source_path': source_path,
                    'file_extension': Path(source_path).suffix,
                    'chunk_size_chars': len(chunk.page_content),
                    'chunk_size_words': len(chunk.page_content.split()),
                    'processing_timestamp': datetime.now().isoformat()
                })

                # Извлекаем заголовок из первых строк или метаданных
                if 'title' not in chunk.metadata:
                    # Берем первые 50 символов как заголовок
                    title = chunk.page_content[:50].replace('\n', ' ').strip()
                    if len(title) < 50 and len(chunk.page_content) > 50:
                        title += "..."
                    chunk.metadata['title'] = title

            all_chunks.extend(doc_chunks)

            # Прогресс
            if (doc_idx + 1) % 10 == 0 or (doc_idx + 1) == len(documents):
                print(f"  Обработано документов: {doc_idx + 1}/{len(documents)}")

        self.stats['total_chunks'] = len(all_chunks)

        # Вычисляем статистику по чанкам
        if all_chunks:
            total_chars = sum(len(chunk.page_content) for chunk in all_chunks)
            total_words = sum(len(chunk.page_content.split()) for chunk in all_chunks)

            avg_chars = total_chars / len(all_chunks)
            avg_words = total_words / len(all_chunks)

            print(f"\n📊 Статистика чанков:")
            print(f"  Всего чанков: {len(all_chunks)}")
            print(f"  Средний размер: {avg_chars:.0f} символов, {avg_words:.0f} слов")
            print(f"  Примерный размер в токенах: {avg_words * 1.3:.0f} (примерно 1.3 токена на слово)")

        return all_chunks

    def create_vector_index(self, chunks: List[Document]) -> Chroma:
        """Создание векторного индекса в ChromaDB"""
        print(f"\n🏗️  Создание векторного индекса...")
        print(f"   Количество чанков: {len(chunks)}")
        print(f"   Это может занять некоторое время...")

        # Создаем векторную базу данных
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory),
            collection_name="knowledge_base",
            collection_metadata={
                "hnsw:space": "cosine",
                "description": "Векторный индекс базы знаний",
                "model": self.stats['model_info']['name'],
                "embedding_dim": str(self.stats['model_info']['embedding_dimensions']),
                "chunk_size": str(self.chunk_size),
                "chunk_overlap": str(self.chunk_overlap),
                "created_at": datetime.now().isoformat(),
                "langchain_version": "1.2.4",
                "chromadb_version": "1.4.1"
            }
        )

        # Сохраняем индекс
        vector_store.persist()

        print(f"✓ Векторный индекс создан и сохранен")
        print(f"  Путь: {self.persist_directory}")

        return vector_store

    def test_index(self, vector_store: Chroma):
        """Тестирование созданного индекса"""
        print(f"\n🧪 Тестирование индекса...")

        test_queries = [
            "архитектура системы",
            "база данных",
            "векторный поиск",
            "машинное обучение",
            "обработка текста"
        ]

        for query in test_queries:
            try:
                print(f"\n  Запрос: '{query}'")
                results = vector_store.similarity_search(query, k=2)

                for i, doc in enumerate(results):
                    source = doc.metadata.get('source_file', 'Неизвестно')
                    chunk_id = doc.metadata.get('chunk_id', 'N/A')
                    preview = doc.page_content[:100].replace('\n', ' ')

                    print(f"    {i+1}. {source} [{chunk_id}]")
                    print(f"       {preview}...")

            except Exception as e:
                print(f"    ⚠️  Ошибка при тестировании: {e}")

    def calculate_index_size(self) -> float:
        """Вычисление размера индекса в МБ"""
        total_size = 0
        for file_path in self.persist_directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size / (1024 * 1024)  # В МБ

    def save_statistics(self):
        """Сохранение статистики индексирования в JSON файл"""
        elapsed_time = self.stats['end_time'] - self.stats['start_time']

        stats_data = {
            "indexing_session": {
                "timestamp": datetime.now().isoformat(),
                "elapsed_time_seconds": round(elapsed_time, 2),
                "elapsed_time_minutes": round(elapsed_time / 60, 2),
                "status": "completed"
            },
            "model": self.stats['model_info'],
            "chunking_config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "separators": ["\\n\\n", "\\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
            },
            "statistics": {
                "total_documents": self.stats['total_documents'],
                "total_chunks": self.stats['total_chunks'],
                "index_size_mb": round(self.calculate_index_size(), 2),
                "chunks_per_second": round(self.stats['total_chunks'] / elapsed_time, 2) if elapsed_time > 0 else 0,
                "documents_per_second": round(self.stats['total_documents'] / elapsed_time, 2) if elapsed_time > 0 else 0
            },
            "system_info": {
                "python_version": "3.12.7",
                "langchain_version": "1.2.4",
                "chromadb_version": "1.4.1",
                "numpy_version": "2.4.1",
                "platform": "Linux"
            },
            "paths": {
                "knowledge_base": str(self.knowledge_base_path),
                "index_directory": str(self.persist_directory)
            }
        }

        # Сохраняем в файл
        with open('indexing_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Статистика сохранена в indexing_statistics.json")

        # Также сохраняем краткую версию для README
        with open('stats_summary.txt', 'w', encoding='utf-8') as f:
            f.write(f"Модель: {self.stats['model_info']['name']}\n")
            f.write(f"Размер эмбеддингов: {self.stats['model_info']['embedding_dimensions']}\n")
            f.write(f"Документов: {self.stats['total_documents']}\n")
            f.write(f"Чанков: {self.stats['total_chunks']}\n")
            f.write(f"Время индексирования: {elapsed_time:.2f} секунд\n")
            f.write(f"Размер индекса: {self.calculate_index_size():.2f} MB\n")

    def run(self) -> Optional[Chroma]:
        """Основной процесс индексирования"""
        print("=" * 70)
        print("СОЗДАНИЕ ВЕКТОРНОГО ИНДЕКСА БАЗЫ ЗНАНИЙ")
        print("=" * 70)

        self.stats['start_time'] = time.time()

        # Проверка окружения
        if not self.validate_environment():
            return None

        # Шаг 1: Загрузка документов
        print(f"\n{'='*50}")
        print("ШАГ 1: ЗАГРУЗКА ДОКУМЕНТОВ")
        print(f"{'='*50}")
        documents = self.load_documents()

        if not documents:
            print("❌ Нет документов для обработки!")
            return None

        # Шаг 2: Разделение на чанки
        print(f"\n{'='*50}")
        print("ШАГ 2: РАЗДЕЛЕНИЕ НА ЧАНКИ")
        print(f"{'='*50}")
        chunks = self.split_documents(documents)

        if not chunks:
            print("❌ Не удалось создать чанки!")
            return None

        # Шаг 3: Создание векторного индекса
        print(f"\n{'='*50}")
        print("ШАГ 3: СОЗДАНИЕ ВЕКТОРНОГО ИНДЕКСА")
        print(f"{'='*50}")
        try:
            vector_store = self.create_vector_index(chunks)
        except Exception as e:
            print(f"❌ Ошибка при создании индекса: {e}")
            return None

        # Шаг 4: Тестирование
        print(f"\n{'='*50}")
        print("ШАГ 4: ТЕСТИРОВАНИЕ")
        print(f"{'='*50}")
        self.test_index(vector_store)

        # Завершение
        self.stats['end_time'] = time.time()

        print(f"\n{'='*50}")
        print("СОХРАНЕНИЕ СТАТИСТИКИ")
        print(f"{'='*50}")
        self.save_statistics()

        # Итоговая информация
        elapsed_time = self.stats['end_time'] - self.stats['start_time']

        print(f"\n{'='*70}")
        print("✅ ИНДЕКСИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print(f"{'='*70}")
        print(f"📊 ИТОГОВАЯ СТАТИСТИКА:")
        print(f"   Время выполнения: {elapsed_time:.2f} секунд ({elapsed_time/60:.2f} минут)")
        print(f"   Обработано документов: {self.stats['total_documents']}")
        print(f"   Создано чанков: {self.stats['total_chunks']}")
        print(f"   Скорость: {self.stats['total_chunks']/elapsed_time:.1f} чанков/сек")
        print(f"   Размер индекса: {self.calculate_index_size():.2f} MB")
        print(f"   Модель: {self.stats['model_info']['name']}")
        print(f"   Размерность: {self.stats['model_info']['embedding_dimensions']}")
        print(f"   Путь к индексу: {self.persist_directory}")
        print(f"{'='*70}")

        return vector_store

def main():
    """Точка входа"""
    # Конфигурация (можно изменить по необходимости)
    config = {
        'knowledge_base_path': 'knowledge_base',
        'persist_directory': 'chroma_db',
        'chunk_size': 800,      # ~500-1000 токенов
        'chunk_overlap': 150    # ~20% перекрытия
    }

    print(f"Конфигурация:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Создаем индексатор и запускаем процесс
    indexer = KnowledgeBaseIndexer(**config)
    vector_store = indexer.run()

    if vector_store:
        print("\n🎉 Готово! Векторный индекс создан успешно.")
        print("   Для поиска используйте query_index.py")
    else:
        print("\n❌ Произошла ошибка при создании индекса")

if __name__ == "__main__":
    main()