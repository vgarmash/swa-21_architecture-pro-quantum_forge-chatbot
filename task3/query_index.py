"""
query_index.py - Поиск по векторному индексу
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class KnowledgeBaseQuery:
    """Класс для поиска по векторному индексу базы знаний"""

    def __init__(self, persist_directory: str = "chroma_db"):
        """
        Инициализация системы поиска

        Args:
            persist_directory: Путь к папке с индексом ChromaDB
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = "knowledge_base"

        if not self.persist_directory.exists():
            print(f"❌ Ошибка: Индекс не найден в {persist_directory}")
            print(f"   Сначала запустите build_index.py для создания индекса")
            sys.exit(1)

        # Загружаем ту же модель эмбеддингов, что использовалась при индексировании
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={'normalize_embeddings': True}
        )

        # Загружаем векторную БД
        self.vector_store = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

        # Получаем информацию о коллекции
        self.collection_info = self._get_collection_info()

        print(f"✅ Индекс загружен из {self.persist_directory}")
        print(f"   Модель: sentence-transformers/all-MiniLM-L6-v2")
        print(f"   Коллекция: {self.collection_name}")
        if self.collection_info:
            print(f"   Количество записей: {self.collection_info.get('count', 'N/A')}")

    def _get_collection_info(self) -> Optional[dict]:
        """Получение информации о коллекции"""
        try:
            collection = self.vector_store._collection
            if collection:
                count = collection.count()
                metadata = collection.metadata or {}
                return {
                    'count': count,
                    'metadata': metadata
                }
        except Exception as e:
            print(f"⚠️  Не удалось получить информацию о коллекции: {e}")
        return None

    def search(self,
               query: str,
               k: int = 5,
               score_threshold: float = 0.5) -> List[Tuple[Document, float]]:
        """
        Поиск релевантных документов по запросу

        Args:
            query: Поисковый запрос
            k: Количество возвращаемых результатов
            score_threshold: Порог сходства (0-1)

        Returns:
            Список кортежей (документ, оценка сходства)
        """
        try:
            # Используем поиск с оценками релевантности
            results = self.vector_store.similarity_search_with_relevance_scores(
                query,
                k=k
            )

            # Фильтруем по порогу
            filtered_results = [
                (doc, score) for doc, score in results
                if score >= score_threshold
            ]

            return filtered_results

        except Exception as e:
            print(f"❌ Ошибка при поиске: {e}")
            return []

    def search_with_filter(self,
                           query: str,
                           filter_dict: dict = None,
                           k: int = 5) -> List[Document]:
        """
        Поиск с фильтрацией по метаданным

        Args:
            query: Поисковый запрос
            filter_dict: Словарь для фильтрации метаданных
            k: Количество результатов

        Returns:
            Список документов
        """
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            return results
        except Exception as e:
            print(f"❌ Ошибка при поиске с фильтром: {e}")
            return []

    def print_results(self, query: str, results: List[Tuple[Document, float]]):
        """
        Красивый вывод результатов поиска

        Args:
            query: Исходный запрос
            results: Результаты поиска
        """
        print(f"\n{'='*80}")
        print(f"🔍 ЗАПРОС: '{query}'")
        print(f"{'='*80}")

        if not results:
            print("❌ По вашему запросу ничего не найдено")
            print("   Попробуйте изменить формулировку или уменьшить порог сходства")
            return

        print(f"✅ Найдено результатов: {len(results)}\n")

        for i, (doc, score) in enumerate(results, 1):
            print(f"{'─'*40}")
            print(f"📄 РЕЗУЛЬТАТ #{i}")
            print(f"{'─'*40}")
            print(f"📊 Сходство: {score:.4f}")
            print(f"📁 Файл: {doc.metadata.get('source_file', 'Неизвестно')}")
            print(f"🆔 ID чанка: {doc.metadata.get('chunk_id', 'N/A')}")
            print(f"📍 Позиция: чанк {doc.metadata.get('chunk_index', 'N/A')} из {doc.metadata.get('total_chunks_in_doc', 'N/A')}")
            print(f"📝 Заголовок: {doc.metadata.get('title', 'Без заголовка')}")

            # Содержимое с ограничением по длине
            content = doc.page_content
            if len(content) > 500:
                content = content[:500] + "..."

            print(f"\n📋 СОДЕРЖИМОЕ:")
            print(f"{'─'*40}")

            # Выводим содержимое с переносами строк
            lines = content.split('\n')
            for line in lines:
                if line.strip():
                    print(f"  {line}")

            print(f"{'─'*40}")
            print(f"🔗 Полный путь: {doc.metadata.get('source_path', 'Неизвестно')}\n")

    def interactive_search(self):
        """Интерактивный режим поиска"""
        print(f"\n{'='*80}")
        print("🎯 ИНТЕРАКТИВНЫЙ ПОИСК ПО БАЗЕ ЗНАНИЙ")
        print(f"{'='*80}")
        print("Команды:")
        print("  /help  - показать эту справку")
        print("  /info  - информация об индексе")
        print("  /k N   - изменить количество результатов (по умолчанию: 5)")
        print("  /th X  - изменить порог сходства (0.0-1.0, по умолчанию: 0.5)")
        print("  /exit  - выход")
        print(f"{'='*80}")

        k = 5
        threshold = 0.5

        while True:
            try:
                user_input = input("\n🎯 Введите запрос или команду: ").strip()

                if not user_input:
                    continue

                # Обработка команд
                if user_input.startswith('/'):
                    if user_input.lower() == '/exit':
                        print("Завершение работы...")
                        break
                    elif user_input.lower() == '/help':
                        print("Команды: /help, /info, /k N, /th X, /exit")
                    elif user_input.lower() == '/info':
                        if self.collection_info:
                            print(f"📊 Информация об индексе:")
                            print(f"   Коллекция: {self.collection_name}")
                            print(f"   Записей: {self.collection_info.get('count', 'N/A')}")
                            metadata = self.collection_info.get('metadata', {})
                            if metadata:
                                print(f"   Метаданные коллекции:")
                                for key, value in metadata.items():
                                    print(f"     {key}: {value}")
                        else:
                            print("Информация о коллекции недоступна")
                    elif user_input.startswith('/k '):
                        try:
                            new_k = int(user_input[3:])
                            if 1 <= new_k <= 20:
                                k = new_k
                                print(f"✅ Количество результатов изменено на: {k}")
                            else:
                                print("❌ Количество должно быть от 1 до 20")
                        except:
                            print("❌ Использование: /k N (где N - число от 1 до 20)")
                    elif user_input.startswith('/th '):
                        try:
                            new_th = float(user_input[4:])
                            if 0.0 <= new_th <= 1.0:
                                threshold = new_th
                                print(f"✅ Порог сходства изменен на: {threshold}")
                            else:
                                print("❌ Порог должен быть от 0.0 до 1.0")
                        except:
                            print("❌ Использование: /th X (где X - число от 0.0 до 1.0)")
                    else:
                        print(f"❌ Неизвестная команда: {user_input}")
                    continue

                # Обычный поисковый запрос
                print(f"\n🔍 Поиск: '{user_input}'...")
                print(f"   Параметры: k={k}, threshold={threshold}")

                results = self.search(user_input, k=k, score_threshold=threshold)
                self.print_results(user_input, results)

            except KeyboardInterrupt:
                print("\n\nЗавершение работы...")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")

def example_queries():
    """Примеры запросов для демонстрации"""
    print(f"\n{'='*80}")
    print("📚 ПРИМЕРЫ ЗАПРОСОВ К БАЗЕ ЗНАНИЙ")
    print(f"{'='*80}")

    query_system = KnowledgeBaseQuery()

    # Примеры запросов
    examples = [
        "Что такое векторный поиск?",
        "Как работает архитектура системы?",
        "Принципы машинного обучения",
        "Базы данных и их типы",
        "Обработка естественного языка",
        "Методы индексирования документов"
    ]

    for query in examples:
        print(f"\n🎯 Пример запроса: '{query}'")
        print(f"{'─'*40}")

        results = query_system.search(query, k=3, score_threshold=0.3)
        query_system.print_results(query, results)

        input("\nНажмите Enter для следующего примера...")

def main():
    """Точка входа"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Поиск по векторному индексу базы знаний',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s --interactive      # Интерактивный режим
  %(prog)s --examples         # Примеры запросов
  %(prog)s --query "текст"    # Один запрос
  %(prog)s --query "текст" --k 3 --threshold 0.7
        """
    )

    parser.add_argument(
        '--mode',
        choices=['interactive', 'examples', 'query'],
        default='interactive',
        help='Режим работы (по умолчанию: interactive)'
    )

    parser.add_argument(
        '--query',
        type=str,
        help='Текст запроса для поиска'
    )

    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Количество возвращаемых результатов (по умолчанию: 5)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Порог сходства (0.0-1.0, по умолчанию: 0.5)'
    )

    parser.add_argument(
        '--index-dir',
        type=str,
        default='chroma_db',
        help='Путь к папке с индексом (по умолчанию: chroma_db)'
    )

    args = parser.parse_args()

    try:
        query_system = KnowledgeBaseQuery(args.index_dir)

        if args.mode == 'interactive':
            query_system.interactive_search()
        elif args.mode == 'examples':
            example_queries()
        elif args.mode == 'query':
            if not args.query:
                print("❌ Для режима 'query' необходимо указать --query")
                parser.print_help()
                return

            print(f"🔍 Поиск запроса: '{args.query}'")
            print(f"   Параметры: k={args.k}, threshold={args.threshold}")

            results = query_system.search(args.query, k=args.k, score_threshold=args.threshold)
            query_system.print_results(args.query, results)

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()