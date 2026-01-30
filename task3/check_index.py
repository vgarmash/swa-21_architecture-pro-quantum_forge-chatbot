"""
check_index.py - Проверка содержимого векторного индекса
"""

import sys
from pathlib import Path

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_chroma import Chroma

try:
    from config import CHROMA_DB_PATH, EMBEDDING_MODEL
except ImportError:
    CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def check_index():
    """Проверка содержимого индекса"""

    index_path = Path(CHROMA_DB_PATH)
    if not index_path.exists():
        print(f"❌ Индекс не найден: {index_path}")
        return

    print(f"🔍 ПРОВЕРКА ИНДЕКСА: {index_path}")
    print("="*60)

    # Загружаем эмбеддинги
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Загружаем векторную БД
    try:
        vector_store = Chroma(
            persist_directory=str(index_path),
            embedding_function=embeddings,
            collection_name="knowledge_base"
        )
        print("✅ Индекс загружен")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return

    # Проверяем коллекцию
    try:
        collection = vector_store._collection

        # Количество записей
        count = collection.count()
        print(f"📊 Всего записей в индексе: {count}")

        # Метаданные
        metadata = collection.metadata or {}
        print(f"📋 Метаданные коллекции:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")

        # Получаем примеры документов
        print(f"\n📄 ПРИМЕРЫ ДОКУМЕНТОВ (первые 10):")
        try:
            results = collection.get(limit=10)

            if results and 'documents' in results:
                for i, doc in enumerate(results['documents']):
                    meta = results['metadatas'][i] if 'metadatas' in results else {}
                    source = meta.get('source_file', meta.get('source', 'unknown'))
                    preview = doc[:200].replace('\n', ' ') + "..." if len(doc) > 200 else doc
                    print(f"\n[{i+1}] {source}")
                    print(f"    ID: {meta.get('chunk_id', 'N/A')}")
                    print(f"    Размер: {len(doc)} chars")
                    print(f"    Preview: {preview}")

        except Exception as e:
            print(f"⚠️  Не удалось получить документы: {e}")

        # Проверяем есть ли документы с ключевыми словами
        print(f"\n🔎 ПОИСК КЛЮЧЕВЫХ СЛОВ В ИНДЕКСЕ:")
        keywords = ["crystal", "blade", "ingredient", "effect", "twin suns", "vitality"]

        for keyword in keywords:
            try:
                results = vector_store.similarity_search_with_score(keyword, k=2)
                if results:
                    print(f"  '{keyword}': найдено {len(results)} результатов")
                    for doc, score in results:
                        source = doc.metadata.get('source_file', 'unknown')
                        print(f"    - {source} (score: {score:.3f})")
                else:
                    print(f"  '{keyword}': не найдено")
            except:
                print(f"  '{keyword}': ошибка поиска")

    except Exception as e:
        print(f"❌ Ошибка при проверке коллекции: {e}")

def test_search():
    """Тестовый поиск"""
    print(f"\n" + "="*60)
    print("🧪 ТЕСТОВЫЙ ПОИСК")
    print("="*60)

    index_path = Path(CHROMA_DB_PATH)
    if not index_path.exists():
        return

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = Chroma(
        persist_directory=str(index_path),
        embedding_function=embeddings,
        collection_name="knowledge_base"
    )

    test_queries = [
        "Crystal Blade",
        "crystal blade",
        "primary ingredient",
        "twin suns alignment",
        "enhanced vitality",
        "What are the main ingredients",
        "effects include",
        "chronic exposure"
    ]

    for query in test_queries:
        print(f"\n🔍 Запрос: '{query}'")
        try:
            results = vector_store.similarity_search_with_score(query, k=3)
            if results:
                for i, (doc, score) in enumerate(results):
                    source = doc.metadata.get('source_file', 'unknown')
                    preview = doc.page_content[:150].replace('\n', ' ')
                    print(f"  {i+1}. {source} (score: {score:.4f})")
                    print(f"     {preview}...")
            else:
                print("  ❌ Не найдено")
        except Exception as e:
            print(f"  ⚠️  Ошибка: {e}")

if __name__ == "__main__":
    check_index()
    test_search()