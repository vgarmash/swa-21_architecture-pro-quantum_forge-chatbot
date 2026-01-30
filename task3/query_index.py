"""
query_index.py - Поиск по векторному индексу базы знаний
Расположен в папке task3
"""

import sys
import argparse
import warnings
from pathlib import Path

# Подавляем предупреждения
warnings.filterwarnings("ignore")


# Импорты
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    EMBEDDING_SOURCE = "langchain_huggingface"
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        EMBEDDING_SOURCE = "langchain_community"
    except ImportError:
        print("❌ Не удалось импортировать HuggingFaceEmbeddings")
        sys.exit(1)

try:
    from langchain_chroma import Chroma
except ImportError:
    print("❌ Не удалось импортировать Chroma")
    sys.exit(1)

from langchain_core.documents import Document
from typing import List, Tuple

try:
    from config import (
        CHROMA_DB_PATH,
        EMBEDDING_MODEL,
        DEFAULT_SEARCH_K,
        DEFAULT_SCORE_THRESHOLD
    )
except ImportError:
    CHROMA_DB_PATH = Path(__file__).parent.parent / "chroma_db"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_SEARCH_K = 10  # Увеличили количество
    DEFAULT_SCORE_THRESHOLD = 0.2  # Уменьшили порог

class KnowledgeBaseQuery:
    """Система поиска по векторному индексу"""

    def __init__(self, index_path: str = None):
        """
        Инициализация

        Args:
            index_path: Путь к папке с индексом
        """
        self.index_path = Path(index_path) if index_path else Path(CHROMA_DB_PATH)

        print(f"🔍 ИНИЦИАЛИЗАЦИЯ ПОИСКА")
        print(f"📁 Путь к индексу: {self.index_path}")
        print(f"📦 Модель: {EMBEDDING_MODEL}")

        # Проверка индекса
        if not self.index_path.exists():
            print(f"\n❌ ОШИБКА: Индекс не найден!")
            print(f"   Запустите: python3 build_index.py")
            sys.exit(1)

        # Инициализация эмбеддингов
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            print(f"❌ Ошибка эмбеддингов: {e}")
            sys.exit(1)

        # Загрузка индекса
        try:
            self.vector_store = Chroma(
                persist_directory=str(self.index_path),
                embedding_function=self.embeddings,
                collection_name="knowledge_base"
            )
            print(f"✅ Индекс загружен")

            # Информация о коллекции
            try:
                count = self.vector_store._collection.count()
                print(f"📊 Записей в индексе: {count}")
            except:
                pass

        except Exception as e:
            print(f"❌ Ошибка загрузки индекса: {e}")
            sys.exit(1)

    def search_with_low_threshold(self,
                                  query: str,
                                  k: int = DEFAULT_SEARCH_K,
                                  threshold: float = 0.1) -> List[Tuple[Document, float]]:
        """
        Поиск с низким порогом для точных ответов

        Args:
            query: Поисковый запрос
            k: Количество результатов
            threshold: Низкий порог сходства

        Returns:
            Список (документ, оценка)
        """
        print(f"\n🔎 ПОИСК: '{query}'")
        print(f"   K: {k}, Порог: {threshold}")

        try:
            # Поиск с низким порогом
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )

            # Фильтрация по низкому порогу
            filtered = [(doc, score) for doc, score in results if score >= threshold]

            print(f"   Найдено: {len(filtered)} результатов")
            return filtered

        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")
            return []

    def extract_answer_from_results(self, query: str, results: List[Tuple[Document, float]]) -> str:
        """Извлечение ответа из результатов"""

        if not results:
            return "❌ Ответ не найден в базе знаний."

        # Ищем наиболее релевантные результаты
        best_results = []
        keywords = ["crystal blade", "primary ingredient", "twin suns", "vitality", "accelerated", "resilience", "chronic exposure"]

        for doc, score in results:
            content_lower = doc.page_content.lower()
            # Проверяем наличие ключевых слов
            keyword_matches = sum(1 for kw in keywords if kw in content_lower)

            # Если есть хотя бы 2 ключевых слова, добавляем в лучшие результаты
            if keyword_matches >= 2:
                best_results.append((doc, score, keyword_matches))

        # Сортируем по количеству совпадений ключевых слов и оценке
        if best_results:
            best_results.sort(key=lambda x: (x[2], x[1]), reverse=True)
            best_doc, best_score, keyword_matches = best_results[0]

            # Извлекаем релевантную часть
            content = best_doc.page_content

            # Ищем релевантные предложения
            import re
            sentences = re.split(r'[.!?]+', content)

            relevant_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(kw in sentence_lower for kw in ["primary ingredient", "effects include", "chronic exposure"]):
                    relevant_sentences.append(sentence.strip())

            if relevant_sentences:
                answer = " ".join(relevant_sentences[:3])  # Берем до 3 релевантных предложений
                source_file = best_doc.metadata.get('source_file', 'unknown')
                return f"📄 Из {source_file}:\n\n{answer}\n\n📊 Сходство: {best_score:.4f}"

        # Если не нашли с ключевыми словами, берем лучший по оценке
        best_doc, best_score = results[0]
        content = best_doc.page_content

        # Обрезаем до разумного размера
        if len(content) > 500:
            content = content[:500] + "..."

        source_file = best_doc.metadata.get('source_file', 'unknown')
        return f"📄 Наиболее релевантный результат из {source_file}:\n\n{content}\n\n📊 Сходство: {best_score:.4f}"

    def print_results(self, query: str, results: List[Tuple[Document, float]]):
        """Вывод результатов"""
        print(f"\n{'='*80}")
        print(f"🔍 РЕЗУЛЬТАТЫ ПОИСКА")
        print(f"📝 Запрос: '{query}'")
        print(f"{'='*80}")

        if not results:
            print("❌ Ничего не найдено")
            return

        print(f"✅ Найдено: {len(results)} результатов\n")

        # Выводим извлеченный ответ
        answer = self.extract_answer_from_results(query, results)
        print("💡 ИЗВЛЕЧЕННЫЙ ОТВЕТ:")
        print("-" * 40)
        print(answer)
        print("-" * 40)

        # Выводим детали для топ-3 результатов
        print(f"\n📋 ДЕТАЛИ ТОП-3 РЕЗУЛЬТАТОВ:")
        for i, (doc, score) in enumerate(results[:3], 1):
            print(f"\n{'─'*40}")
            print(f"📄 РЕЗУЛЬТАТ #{i}")
            print(f"{'─'*40}")
            print(f"📊 Сходство: {score:.4f} ({score*100:.1f}%)")
            print(f"📁 Файл: {doc.metadata.get('source_file', 'unknown')}")

            content = doc.page_content.strip()
            if len(content) > 300:
                content = content[:300] + "..."

            print(f"\n📋 Содержимое:")
            print(f"{'-'*30}")
            print(content)
            print(f"{'-'*30}")

def main():
    """Точка входа"""
    parser = argparse.ArgumentParser(
        description='Поиск по векторному индексу базы знаний',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Примеры:
  {sys.argv[0]} --query "Crystal Blade ingredients and effects"
  {sys.argv[0]} --query "twin suns alignment" --k 15
  {sys.argv[0]} --query "primary ingredient" --threshold 0.1

Параметры по умолчанию:
  Индекс: {CHROMA_DB_PATH}
  Модель: {EMBEDDING_MODEL}
  K: {DEFAULT_SEARCH_K}
  Порог: {DEFAULT_SCORE_THRESHOLD}
        """
    )

    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Текст запроса'
    )

    parser.add_argument(
        '--k',
        type=int,
        default=DEFAULT_SEARCH_K,
        help=f'Количество результатов (по умолчанию: {DEFAULT_SEARCH_K})'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,  # НИЗКИЙ ПОРОГ для точных ответов
        help=f'Порог сходства (по умолчанию: 0.1)'
    )

    parser.add_argument(
        '--index-dir',
        type=str,
        help=f'Путь к папке с индексом (по умолчанию: {CHROMA_DB_PATH})'
    )

    args = parser.parse_args()

    try:
        # Создаем объект поиска
        query_system = KnowledgeBaseQuery(args.index_dir)

        # Выполняем поиск
        results = query_system.search_with_low_threshold(
            args.query,
            k=args.k,
            threshold=args.threshold
        )

        # Выводим результаты
        query_system.print_results(args.query, results)

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()