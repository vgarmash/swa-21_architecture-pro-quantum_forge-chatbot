# test_rag.py
"""
Тестирование RAG системы с использованием вопросов из JSONL файла
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
import time

# Добавляем родительскую директорию в путь для импорта
sys.path.append(str(Path(__file__).parent))

from core.rag_pipeline import RAGPipeline
from config import CONFIG, load_few_shot_examples, FEW_SHOT_FILE_PATH # Добавлены константы!


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGTester:
    """Класс для тестирования RAG системы"""

    def __init__(self, config=None):
        """
        Инициализация тестера

        Args:
            config: Конфигурация RAG
        """
        self.config = config or CONFIG
        self.pipeline = RAGPipeline(self.config)
        self.test_results = []

    def load_test_questions(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Загрузка тестовых вопросов из JSONL файла

        Args:
            file_path: Путь к JSONL файлу

        Returns:
            List[Dict]: Список тестовых вопросов
        """
        try:
            examples = load_few_shot_examples(file_path)
            logger.info(f"Загружено {len(examples)} тестовых вопросов из {file_path}")
            return examples
        except Exception as e:
            logger.error(f"Ошибка загрузки тестовых вопросов: {e}")
            return []

    def test_single_question(
            self,
            question: str,
            expected_answer: str = None,
            max_wait_time: int = 30
    ) -> Dict[str, Any]:
        """
        Тестирование одного вопроса

        Args:
            question: Вопрос для тестирования
            expected_answer: Ожидаемый ответ (опционально)
            max_wait_time: Максимальное время ожидания в секундах

        Returns:
            Dict: Результаты теста
        """
        start_time = time.time()

        try:
            logger.info(f"Тестируем вопрос: {question[:100]}...")

            # Обработка вопроса через RAG
            response = self.pipeline.process_query(question)

            # Анализ ответа
            processing_time = time.time() - start_time

            result = {
                "question": question,
                "answer": response.answer,
                "expected_answer": expected_answer,
                "processing_time": processing_time,
                "has_context": response.has_context,
                "documents_found": len(response.retrieved_docs),
                "sources": response.used_sources,
                "success": True,
                "error": None
            }

            # Проверка на превышение времени
            if processing_time > max_wait_time:
                result["success"] = False
                result["error"] = f"Превышено время ожидания ({max_wait_time} сек)"

            logger.info(f"  Ответ получен за {processing_time:.2f} сек, документов: {len(response.retrieved_docs)}")

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"  Ошибка при обработке: {e}")

            return {
                "question": question,
                "answer": None,
                "expected_answer": expected_answer,
                "processing_time": processing_time,
                "has_context": False,
                "documents_found": 0,
                "sources": [],
                "success": False,
                "error": str(e)
            }

    def test_from_file(
            self,
            file_path: Path,
            max_questions: int = None,
            categories: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Тестирование на основе вопросов из файла

        Args:
            file_path: Путь к JSONL файлу с вопросами
            max_questions: Максимальное количество вопросов для тестирования
            categories: Фильтр по категориям

        Returns:
            List[Dict]: Результаты тестирования
        """
        # Загрузка вопросов
        test_examples = self.load_test_questions(file_path)

        if not test_examples:
            logger.error("Не удалось загрузить тестовые вопросы")
            return []

        # Фильтрация по категориям, если указаны
        if categories:
            test_examples = [
                ex for ex in test_examples
                if ex.get('category', '').lower() in [c.lower() for c in categories]
            ]
            logger.info(f"После фильтрации по категориям осталось {len(test_examples)} вопросов")

        # Ограничение количества вопросов
        if max_questions and len(test_examples) > max_questions:
            test_examples = test_examples[:max_questions]
            logger.info(f"Ограничение до {max_questions} вопросов")

        # Тестирование каждого вопроса
        results = []
        total_questions = len(test_examples)

        logger.info(f"Начинаем тестирование {total_questions} вопросов...")

        for i, example in enumerate(test_examples, 1):
            logger.info(f"\nВопрос {i}/{total_questions}")

            result = self.test_single_question(
                question=example['question'],
                expected_answer=example.get('answer')
            )

            results.append(result)

            # Небольшая пауза между запросами
            if i < total_questions:
                time.sleep(1)

        return results

    def print_summary(self, results: List[Dict[str, Any]]):
        """
        Вывод сводки по результатам тестирования

        Args:
            results: Результаты тестирования
        """
        if not results:
            print("Нет результатов для анализа")
            return

        total = len(results)
        successful = sum(1 for r in results if r['success'])
        with_context = sum(1 for r in results if r['has_context'])

        avg_time = sum(r['processing_time'] for r in results) / total
        avg_docs = sum(r['documents_found'] for r in results) / total

        print("\n" + "="*80)
        print("СВОДКА ТЕСТИРОВАНИЯ RAG СИСТЕМЫ")
        print("="*80)
        print(f"Всего вопросов: {total}")
        print(f"Успешно обработано: {successful} ({successful/total*100:.1f}%)")
        print(f"Ответы с контекстом: {with_context} ({with_context/total*100:.1f}%)")
        print(f"Среднее время ответа: {avg_time:.2f} сек")
        print(f"Среднее найденных документов: {avg_docs:.1f}")

        # Анализ ошибок
        errors = [r for r in results if not r['success']]
        if errors:
            print(f"\nОшибки ({len(errors)}):")
            for error in errors[:5]:  # Показываем первые 5 ошибок
                print(f"  - {error['question'][:50]}...: {error['error']}")

        # Примеры успешных ответов
        successes = [r for r in results if r['success'] and r['has_context']]
        if successes:
            print(f"\nПримеры успешных ответов:")
            for success in successes[:3]:  # Показываем первые 3 успеха
                answer_preview = success['answer'][:100] + "..." if len(success['answer']) > 100 else success['answer']
                print(f"  Вопрос: {success['question'][:50]}...")
                print(f"  Ответ: {answer_preview}")
                print(f"  Документов: {success['documents_found']}, Время: {success['processing_time']:.2f} сек")
                print()

        print("="*80)

    def save_results(self, results: List[Dict[str, Any]], output_file: Path):
        """
        Сохранение результатов тестирования в JSON файл

        Args:
            results: Результаты тестирования
            output_file: Путь для сохранения результатов
        """
        try:
            # Преобразуем объекты для сериализации
            serializable_results = []
            for result in results:
                serializable_result = result.copy()
                # Преобразуем Path в строку если есть
                if 'sources' in serializable_result:
                    serializable_result['sources'] = list(serializable_result['sources'])
                serializable_results.append(serializable_result)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)

            logger.info(f"Результаты сохранены в {output_file}")

        except Exception as e:
            logger.error(f"Ошибка сохранения результатов: {e}")


def main():
    """Основная функция тестирования"""
    import argparse

    parser = argparse.ArgumentParser(description='Тестирование RAG системы')

    parser.add_argument(
        '--test-file',
        type=Path,
        default=FEW_SHOT_FILE_PATH,
        help='Путь к JSONL файлу с тестовыми вопросами'
    )

    parser.add_argument(
        '--max-questions',
        type=int,
        default=10,
        help='Максимальное количество вопросов для тестирования'
    )

    parser.add_argument(
        '--categories',
        nargs='+',
        help='Фильтр по категориям (например: magic_items characters)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path("test_results.json"),
        help='Путь для сохранения результатов тестирования'
    )

    parser.add_argument(
        '--config-file',
        type=Path,
        help='Путь к файлу конфигурации (опционально)'
    )

    args = parser.parse_args()

    # Проверка существования тестового файла
    if not args.test_file.exists():
        print(f"Ошибка: Тестовый файл не найден: {args.test_file}")
        print(f"Ожидаемый путь: {args.test_file.absolute()}")
        sys.exit(1)

    print(f"Тестовый файл: {args.test_file}")
    print(f"Максимум вопросов: {args.max_questions}")
    if args.categories:
        print(f"Категории: {', '.join(args.categories)}")

    try:
        # Создаем тестер
        tester = RAGTester()

        # Получаем информацию о few-shot примерах
        pipeline_info = tester.pipeline.get_pipeline_info()
        print(f"\nИспользуется few-shot примеров: {pipeline_info.get('few_shot_examples', 'N/A')}")

        # Выполняем тестирование
        results = tester.test_from_file(
            file_path=args.test_file,
            max_questions=args.max_questions,
            categories=args.categories
        )

        # Выводим сводку
        tester.print_summary(results)

        # Сохраняем результаты
        if results:
            tester.save_results(results, args.output)

        print(f"\nТестирование завершено!")

    except Exception as e:
        logger.error(f"Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()