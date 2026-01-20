"""
build_index.py - Создание векторного индекса базы знаний
Версия для LangChain 1.x с актуальными версиями
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Добавляем путь для импорта config
sys.path.append(str(Path(__file__).parent))

print("=" * 60)
print("ПРОВЕРКА ИМПОРТОВ...")
print("=" * 60)

# Импорты LangChain 1.x с проверкой
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("✓ langchain_text_splitters импортирован")
except ImportError as e:
    print(f"❌ langchain_text_splitters: {e}")
    print("Установите: pip install langchain-text-splitters==1.1.0")
    sys.exit(1)

try:
    from langchain_community.document_loaders import (
        DirectoryLoader,
        TextLoader,
        UnstructuredMarkdownLoader,
        PyPDFLoader
    )
    print("✓ langchain_community.document_loaders импортирован")
except ImportError as e:
    print(f"❌ langchain_community.document_loaders: {e}")
    print("Установите: pip install langchain-community==0.4.1")
    sys.exit(1)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✓ langchain_huggingface импортирован")
    EMBEDDING_SOURCE = "langchain_huggingface"
except ImportError as e:
    print(f"⚠️  langchain_huggingface: {e}")
    print("Используем langchain_community.embeddings...")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("✓ langchain_community.embeddings импортирован")
        EMBEDDING_SOURCE = "langchain_community"
    except ImportError as e2:
        print(f"❌ langchain_community.embeddings: {e2}")
        print("Установите: pip install langchain-huggingface==1.2.0 или langchain-community==0.4.1")
        sys.exit(1)

try:
    from langchain_chroma import Chroma
    print("✓ langchain_chroma импортирован")
except ImportError as e:
    print(f"❌ langchain_chroma: {e}")
    print("Установите: pip install langchain-chroma==1.1.0")
    sys.exit(1)

try:
    from langchain_core.documents import Document
    print("✓ langchain_core.documents импортирован")
except ImportError as e:
    print(f"❌ langchain_core.documents: {e}")
    sys.exit(1)

print("=" * 60)
print("ВСЕ ИМПОРТЫ УСПЕШНЫ")
print("=" * 60)

# Импорт конфигурации
try:
    from config import (
        KNOWLEDGE_BASE_PATH,
        CHROMA_DB_PATH,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSIONS,
        SUPPORTED_EXTENSIONS
    )
    print("✓ Конфигурация загружена из config.py")
except ImportError as e:
    print(f"⚠️  config.py не найден, используются значения по умолчанию: {e}")
    # Значения по умолчанию
    PROJECT_ROOT = Path(__file__).parent.parent
    KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base"
    CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSIONS = 384
    SUPPORTED_EXTENSIONS = {'.txt': 'text', '.md': 'markdown', '.pdf': 'pdf'}

class KnowledgeBaseIndexer:
    """
    Класс для создания векторного индекса базы знаний
    """

    def __init__(self):
        """Инициализация индексатора"""

        # Пути
        self.knowledge_base_path = Path(KNOWLEDGE_BASE_PATH).resolve()
        self.chroma_db_path = Path(CHROMA_DB_PATH).resolve()

        print(f"\n📁 Конфигурация путей:")
        print(f"   База знаний: {self.knowledge_base_path}")
        print(f"   Индекс: {self.chroma_db_path}")

        # Статистика
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_documents': 0,
            'total_chunks': 0,
            'model_info': {
                'name': EMBEDDING_MODEL,
                'embedding_dimensions': EMBEDDING_DIMENSIONS,
                'url': f'https://huggingface.co/{EMBEDDING_MODEL}'
            },
            'chunking': {
                'chunk_size': CHUNK_SIZE,
                'chunk_overlap': CHUNK_OVERLAP
            }
        }

        # Инициализация компонентов
        self._initialize_components()

    def _initialize_components(self):
        """Инициализация компонентов LangChain"""
        print(f"\n🔧 Инициализация компонентов...")

        # Модель эмбеддингов с разными параметрами для разных источников
        if EMBEDDING_SOURCE == "langchain_huggingface":
            # Для langchain-huggingface 1.2.0
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    # В новой версии параметр может называться по-другому или отсутствовать
                }
            )
        else:
            # Для langchain_community
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'show_progress_bar': False
                }
            )

        print(f"   Модель эмбеддингов: {EMBEDDING_MODEL}")
        print(f"   Источник: {EMBEDDING_SOURCE}")

        # Текстовый сплиттер
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            keep_separator=False,
            add_start_index=True
        )
        print(f"   Размер чанка: {CHUNK_SIZE} символов")
        print(f"   Перекрытие: {CHUNK_OVERLAP} символов")

    def validate_environment(self) -> bool:
        """Проверка окружения"""
        print(f"\n🔍 Проверка окружения...")

        if not self.knowledge_base_path.exists():
            print(f"❌ Папка '{self.knowledge_base_path}' не найдена!")
            return False

        # Создаем папку для индекса
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)

        print(f"✓ База знаний: {self.knowledge_base_path}")
        print(f"✓ Папка для индекса: {self.chroma_db_path}")
        return True

    def load_documents(self) -> List[Document]:
        """Загрузка документов"""
        print(f"\n📂 Загрузка документов...")

        documents = []

        # Маппинг форматов на загрузчики
        loaders = {
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.pdf': PyPDFLoader,
        }

        for ext, loader_class in loaders.items():
            try:
                pattern = f"**/*{ext}"
                loader = DirectoryLoader(
                    str(self.knowledge_base_path),
                    glob=pattern,
                    loader_cls=loader_class,
                    show_progress=False,
                    use_multithreading=True,
                    silent_errors=True
                )

                loaded = loader.load()
                if loaded:
                    documents.extend(loaded)
                    print(f"  ✓ {ext}: {len(loaded)} файлов")
            except Exception as e:
                print(f"  ⚠️  {ext}: {str(e)[:100]}...")

        self.stats['total_documents'] = len(documents)

        if not documents:
            print("⚠️  Документы не найдены!")

        print(f"  Всего: {len(documents)} документов")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Разделение на чанки"""
        print(f"\n✂️  Разделение на чанки...")

        all_chunks = []

        for doc_idx, doc in enumerate(documents):
            try:
                doc_chunks = self.text_splitter.split_documents([doc])

                for chunk_idx, chunk in enumerate(doc_chunks):
                    # Метаданные
                    source_path = chunk.metadata.get('source', 'unknown')

                    chunk.metadata.update({
                        'chunk_id': hashlib.md5(
                            f"{source_path}_{doc_idx}_{chunk_idx}".encode()
                        ).hexdigest()[:12],
                        'chunk_index': chunk_idx,
                        'total_chunks_in_doc': len(doc_chunks),
                        'document_index': doc_idx,
                        'source_file': Path(source_path).name,
                        'source_path': str(Path(source_path)),
                        'file_type': Path(source_path).suffix,
                        'chunk_size_chars': len(chunk.page_content),
                        'chunk_size_words': len(chunk.page_content.split()),
                        'timestamp': datetime.now().isoformat()
                    })

                all_chunks.extend(doc_chunks)

                if (doc_idx + 1) % 5 == 0 or (doc_idx + 1) == len(documents):
                    print(f"  Обработано: {doc_idx + 1}/{len(documents)}")

            except Exception as e:
                print(f"  ⚠️  Ошибка в документе {doc_idx}: {str(e)[:100]}...")

        self.stats['total_chunks'] = len(all_chunks)

        if all_chunks:
            avg_size = sum(len(c.page_content) for c in all_chunks) / len(all_chunks)
            print(f"✓ Чанков: {len(all_chunks)}")
            print(f"✓ Средний размер: {avg_size:.0f} символов")

        return all_chunks

    def create_vector_index(self, chunks: List[Document]) -> Chroma:
        """Создание векторного индекса"""
        print(f"\n🏗️  Создание векторного индекса...")

        try:
            # Создаем индекс
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(self.chroma_db_path),
                collection_name="knowledge_base",
                collection_metadata={
                    "hnsw:space": "cosine",
                    "model": EMBEDDING_MODEL,
                    "embedding_dim": str(EMBEDDING_DIMENSIONS),
                    "created_at": datetime.now().isoformat(),
                    "source": EMBEDDING_SOURCE
                }
            )

            print(f"✓ Индекс создан")
            print(f"   Путь: {self.chroma_db_path}")
            print(f"   Чанков: {len(chunks)}")
            return vector_store

        except Exception as e:
            print(f"❌ Ошибка создания индекса: {e}")
            raise

    def run(self) -> Optional[Chroma]:
        """Запуск индексации"""
        print("\n" + "="*70)
        print("ЗАПУСК ИНДЕКСИРОВАНИЯ")
        print("="*70)

        self.stats['start_time'] = time.time()

        # Проверка
        if not self.validate_environment():
            return None

        # Загрузка
        documents = self.load_documents()
        if not documents:
            print("\n❌ Нет документов для обработки")
            return None

        # Разделение
        chunks = self.split_documents(documents)
        if not chunks:
            print("\n❌ Не удалось создать чанки")
            return None

        # Создание индекса
        try:
            vector_store = self.create_vector_index(chunks)
        except Exception as e:
            print(f"\n❌ Ошибка создания индекса: {e}")
            return None

        # Завершение
        self.stats['end_time'] = time.time()

        # Статистика
        elapsed = self.stats['end_time'] - self.stats['start_time']

        print(f"\n" + "="*70)
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print("="*70)
        print(f"   Время: {elapsed:.2f} секунд")
        print(f"   Документы: {self.stats['total_documents']}")
        print(f"   Чанки: {self.stats['total_chunks']}")
        if elapsed > 0:
            print(f"   Скорость: {self.stats['total_chunks']/elapsed:.1f} чанков/сек")
        print(f"   Модель: {self.stats['model_info']['name']}")
        print(f"   Размерность: {self.stats['model_info']['embedding_dimensions']}")
        print("="*70)

        return vector_store

def main():
    """Точка входа"""
    print("Задание 3: Создание векторного индекса базы знаний")
    print(f"Версии: LangChain 1.2.4, ChromaDB 1.4.1, LangChain-HuggingFace 1.2.0")

    indexer = KnowledgeBaseIndexer()
    result = indexer.run()

    if result:
        print("\n✅ Задание выполнено успешно!")
        print(f"   Индекс: {CHROMA_DB_PATH}")
        print(f"\n📊 Для поиска запустите:")
        print(f"   python3 query_index.py --mode interactive")
    else:
        print("\n❌ Задание не выполнено")

if __name__ == "__main__":
    main()