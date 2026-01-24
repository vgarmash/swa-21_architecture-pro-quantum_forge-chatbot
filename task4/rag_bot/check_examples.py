# check_examples.py
"""
Скрипт для проверки загрузки few-shot примеров из файла
"""

import sys
import json
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Добавляем родительскую директорию в путь
sys.path.append(str(Path(__file__).parent))

from config import (
    FEW_SHOT_FILE_PATH,
    GENERATED_PATH,
    PROJECT_ROOT,
    load_few_shot_examples,
    select_best_examples,
    get_default_fallback_examples
)


def load_and_analyze_examples(file_path: Path) -> dict:
    """
    Загрузка и анализ few-shot примеров

    Args:
        file_path: Путь к JSONL файлу

    Returns:
        dict: Результаты анализа
    """
    print(f"Проверка файла: {file_path}")
    print(f"Абсолютный путь: {file_path.absolute()}")
    print(f"Размер файла: {file_path.stat().st_size:,} байт" if file_path.exists() else "Файл не существует")
    print("-" * 80)

    if not file_path.exists():
        print(f"⚠️ Файл не найден: {file_path}")
        print(f"Директория {GENERATED_PATH} существует: {GENERATED_PATH.exists()}")

        # Показываем содержимое директории
        if GENERATED_PATH.exists():
            print(f"\nСодержимое директории {GENERATED_PATH}:")
            for item in GENERATED_PATH.iterdir():
                print(f"  - {item.name} ({'файл' if item.is_file() else 'папка'})")

        return {
            "file_exists": False,
            "total_examples": 0,
            "examples": [],
            "categories": {},
            "difficulties": {}
        }

    try:
        # Загружаем все примеры
        all_examples = load_few_shot_examples(file_path)
        print(f"✓ Загружено примеров: {len(all_examples)}")

        if not all_examples:
            print("⚠️ Файл пуст или содержит невалидные данные")
            return {
                "file_exists": True,
                "total_examples": 0,
                "examples": [],
                "categories": {},
                "difficulties": {}
            }

        # Анализ категорий и сложности
        categories = {}
        difficulties = {}
        context_required_count = 0

        for example in all_examples:
            category = example.get('category', 'unknown')
            difficulty = example.get('difficulty', 'unknown')

            categories[category] = categories.get(category, 0) + 1
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

            if example.get('context_required', True):
                context_required_count += 1

        # Анализ структуры примеров
        print("\n📊 Распределение по категориям:")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_examples)) * 100
            print(f"  {category:20} {count:3} примеров ({percentage:.1f}%)")

        print("\n📊 Распределение по сложности:")
        for difficulty, count in sorted(difficulties.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_examples)) * 100
            print(f"  {difficulty:20} {count:3} примеров ({percentage:.1f}%)")

        print(f"\n📊 Контекст требуется: {context_required_count} из {len(all_examples)} "
              f"({context_required_count/len(all_examples)*100:.1f}%)")

        # Проверка качества данных
        print("\n🔍 Проверка качества данных:")

        # Проверка наличия ключевых полей
        required_fields = ['question', 'answer']
        optional_fields = ['source_docs', 'difficulty', 'category', 'context_required']

        for field in required_fields:
            missing_count = sum(1 for ex in all_examples if field not in ex or not ex[field])
            if missing_count > 0:
                print(f"  ⚠️  Отсутствует поле '{field}': {missing_count} примеров")

        # Проверка длины вопросов и ответов
        short_questions = sum(1 for ex in all_examples if len(ex.get('question', '')) < 5)
        short_answers = sum(1 for ex in all_examples if len(ex.get('answer', '')) < 10)

        if short_questions > 0:
            print(f"  ⚠️  Слишком короткие вопросы (<5 символов): {short_questions} примеров")

        if short_answers > 0:
            print(f"  ⚠️  Слишком короткие ответы (<10 символов): {short_answers} примеров")

        return {
            "file_exists": True,
            "total_examples": len(all_examples),
            "examples": all_examples,
            "categories": categories,
            "difficulties": difficulties,
            "context_required_count": context_required_count
        }

    except Exception as e:
        print(f"❌ Ошибка при загрузке файла: {e}")
        import traceback
        traceback.print_exc()
        return {
            "file_exists": True,
            "error": str(e),
            "total_examples": 0,
            "examples": [],
            "categories": {},
            "difficulties": {}
        }


def test_selection_strategies(all_examples: list, analysis_result: dict):
    """
    Тестирование различных стратегий отбора примеров

    Args:
        all_examples: Все загруженные примеры
        analysis_result: Результаты анализа
    """
    if not all_examples:
        print("❌ Нет примеров для тестирования стратегий отбора")
        return

    print("\n" + "="*80)
    print("ТЕСТИРОВАНИЕ СТРАТЕГИЙ ОТБОРА ПРИМЕРОВ")
    print("="*80)

    strategies = [
        {
            "name": "Без фильтров (по умолчанию)",
            "kwargs": {"count": 2}
        },
        {
            "name": "Только легкие примеры",
            "kwargs": {"count": 2, "difficulty_filter": "easy"}
        },
        {
            "name": "Только средняя сложность",
            "kwargs": {"count": 2, "difficulty_filter": "medium"}
        },
        {
            "name": "Категория magic_items",
            "kwargs": {"count": 2, "category_filter": "magic_items"}
        },
        {
            "name": "Категория characters",
            "kwargs": {"count": 2, "category_filter": "characters"}
        },
        {
            "name": "Больше примеров (5)",
            "kwargs": {"count": 5}
        },
        {
            "name": "Без приоритета контекста",
            "kwargs": {"count": 2, "prioritize_with_context": False}
        }
    ]

    for strategy in strategies:
        print(f"\n📋 Стратегия: {strategy['name']}")

        try:
            selected = select_best_examples(all_examples, **strategy['kwargs'])
            print(f"  Отобрано: {len(selected)} примеров")

            if selected:
                for i, ex in enumerate(selected, 1):
                    question_preview = ex['question'][:50] + "..." if len(ex['question']) > 50 else ex['question']
                    print(f"    {i}. {question_preview}")
                    print(f"       Категория: {ex.get('category', 'N/A')}, "
                          f"Сложность: {ex.get('difficulty', 'N/A')}")
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")


def display_sample_examples(all_examples: list, count: int = 3):
    """
    Отображение примеров для проверки

    Args:
        all_examples: Все загруженные примеры
        count: Количество примеров для отображения
    """
    if not all_examples:
        return

    print("\n" + "="*80)
    print(f"ПРИМЕРЫ ДАННЫХ (первые {count})")
    print("="*80)

    for i, example in enumerate(all_examples[:count], 1):
        print(f"\n📄 Пример {i}:")
        print(f"   Вопрос: {example['question']}")

        answer = example['answer']
        if len(answer) > 100:
            answer = answer[:100] + "..."
        print(f"   Ответ: {answer}")

        print(f"   Категория: {example.get('category', 'N/A')}")
        print(f"   Сложность: {example.get('difficulty', 'N/A')}")

        source_docs = example.get('source_docs', [])
        if source_docs:
            print(f"   Источники: {', '.join(source_docs)}")

        context_required = example.get('context_required', True)
        print(f"   Требуется контекст: {'Да' if context_required else 'Нет'}")

        metadata = example.get('metadata', {})
        if metadata:
            print(f"   Метаданные: {metadata}")


def check_fallback_examples():
    """Проверка fallback примеров"""
    print("\n" + "="*80)
    print("ПРОВЕРКА FALLBACK ПРИМЕРОВ")
    print("="*80)

    fallback_examples = get_default_fallback_examples()
    print(f"Количество fallback примеров: {len(fallback_examples)}")

    for i, example in enumerate(fallback_examples, 1):
        print(f"\nFallback пример {i}:")
        print(f"  Вопрос: {example['question']}")
        print(f"  Ответ: {example['answer'][:80]}..." if len(example['answer']) > 80 else f"  Ответ: {example['answer']}")
        print(f"  Категория: {example.get('category', 'N/A')}")
        print(f"  Источники: {', '.join(example.get('source_docs', []))}")


def export_examples_summary(analysis_result: dict, output_file: Path):
    """
    Экспорт сводки по примерам в файл

    Args:
        analysis_result: Результаты анализа
        output_file: Путь для сохранения
    """
    if not analysis_result.get("examples"):
        print("❌ Нет данных для экспорта")
        return

    summary = {
        "total_examples": analysis_result["total_examples"],
        "categories": analysis_result["categories"],
        "difficulties": analysis_result["difficulties"],
        "context_required_count": analysis_result.get("context_required_count", 0),
        "sample_examples": [
            {
                "question": ex["question"],
                "answer_preview": ex["answer"][:100] + "..." if len(ex["answer"]) > 100 else ex["answer"],
                "category": ex.get("category"),
                "difficulty": ex.get("difficulty")
            }
            for ex in analysis_result["examples"][:5]
        ]
    }

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Сводка экспортирована в: {output_file}")
    except Exception as e:
        print(f"❌ Ошибка при экспорте: {e}")


def main():
    """Основная функция проверки"""
    import argparse

    parser = argparse.ArgumentParser(description='Проверка few-shot примеров')

    parser.add_argument(
        '--file',
        type=Path,
        default=FEW_SHOT_FILE_PATH,
        help='Путь к JSONL файлу с few-shot примерами'
    )

    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Показать все примеры (может быть много)'
    )

    parser.add_argument(
        '--export',
        type=Path,
        help='Путь для экспорта сводки в JSON'
    )

    parser.add_argument(
        '--check-fallback',
        action='store_true',
        help='Проверить fallback примеры'
    )

    args = parser.parse_args()

    print(f"Корневая директория проекта: {PROJECT_ROOT}")
    print(f"Директория generated: {GENERATED_PATH}")
    print(f"Существует: {GENERATED_PATH.exists()}")
    print()

    # Загружаем и анализируем примеры
    analysis_result = load_and_analyze_examples(args.file)

    # Если есть примеры, продолжаем анализ
    if analysis_result.get("total_examples", 0) > 0:
        all_examples = analysis_result["examples"]

        # Тестируем стратегии отбора
        test_selection_strategies(all_examples, analysis_result)

        # Показываем примеры
        if args.show_all:
            display_sample_examples(all_examples, len(all_examples))
        else:
            display_sample_examples(all_examples, 3)

        # Экспорт результатов
        if args.export:
            export_examples_summary(analysis_result, args.export)

    # Проверка fallback примеров
    if args.check_fallback:
        check_fallback_examples()

    print("\n" + "="*80)

    if analysis_result.get("file_exists"):
        if analysis_result.get("total_examples", 0) > 0:
            print("✓ Проверка завершена успешно!")
            print(f"   Загружено примеров: {analysis_result['total_examples']}")
            print(f"   Категорий: {len(analysis_result['categories'])}")
            print(f"   Уровней сложности: {len(analysis_result['difficulties'])}")
        else:
            print("⚠️ Файл существует, но примеры не загружены")
    else:
        print("❌ Файл не найден. Убедитесь, что:")
        print(f"   1. Файл существует: {args.file}")
        print(f"   2. Директория {GENERATED_PATH} существует")
        print(f"   3. У вас есть права на чтение файла")

    print("="*80)


if __name__ == "__main__":
    main()