# main.py
"""
Основной файл запуска RAG-бота
"""

import sys
import argparse
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent))

from interfaces.repl_interface import main as run_repl
from interfaces.telegram_bot import main as run_telegram_bot
from config import create_config, CONFIG, PROJECT_ROOT, DEFAULT_SEARCH_K, DEFAULT_SCORE_THRESHOLD


def main():
    """
    Основная точка входа в приложение
    """
    parser = argparse.ArgumentParser(
        description='RAG Бот - система вопросов и ответов на основе знаний',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --mode repl
  python main.py --mode telegram --telegram-token "YOUR_TOKEN"
  python main.py --mode test --max-questions 5
  python main.py --vector-db custom_chroma_db --few-shot-file my_examples.jsonl
        """
    )

    parser.add_argument(
        '--mode',
        choices=['repl', 'telegram', 'test'],
        default='repl',
        help='Режим работы (repl: консоль, telegram: телеграм бот, test: тестирование)'
    )

    parser.add_argument(
        '--telegram-token',
        help='Токен для Telegram бота (требуется для режима telegram)'
    )

    parser.add_argument(
        '--vector-db',
        type=Path,
        default=None,
        help=f'Путь к векторной базе данных (по умолчанию: {CONFIG.vector_store.persist_directory.relative_to(PROJECT_ROOT)})'
    )

    parser.add_argument(
        '--collection',
        type=str,
        default=None,
        help=f'Название коллекции (по умолчанию: {CONFIG.vector_store.collection_name})'
    )

    parser.add_argument(
        '--few-shot-file',
        type=Path,
        default=None,
        help=f'Путь к файлу с few-shot примерами (по умолчанию: {CONFIG.few_shot.examples_file.relative_to(PROJECT_ROOT)})'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help=f'Название модели LLM (по умолчанию: {CONFIG.llm.model_name})'
    )

    parser.add_argument(
        '--search-k',
        type=int,
        default=None,
        help=f'Количество документов для поиска (по умолчанию: {DEFAULT_SEARCH_K})'  # Исправлено!
    )

    parser.add_argument(
        '--score-threshold',
        type=float,
        default=None,
        help=f'Порог схожести документов (по умолчанию: {DEFAULT_SCORE_THRESHOLD})'  # Исправлено!
    )

    parser.add_argument(
        '--examples-to-use',
        type=int,
        default=None,
        help=f'Количество few-shot примеров для использования (по умолчанию: {CONFIG.few_shot.examples_to_use})'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default=None,
        help='Уровень логирования'
    )

    parser.add_argument(
        '--max-questions',
        type=int,
        default=10,
        help='Максимальное количество вопросов для тестирования (только для режима test)'
    )

    parser.add_argument(
        '--categories',
        nargs='+',
        help='Фильтр по категориям для тестирования (только для режима test)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path("test_results.json"),
        help='Путь для сохранения результатов тестирования (только для режима test)'
    )

    args = parser.parse_args()

    # Создаем конфигурацию на основе аргументов
    config = create_config(
        vector_store_dir=args.vector_db,
        collection_name=args.collection,
        model_name=args.model,
        few_shot_file=args.few_shot_file,
        examples_to_use=args.examples_to_use,
        search_k=args.search_k,
        score_threshold=args.score_threshold,
        log_level=args.log_level
    )

    # Переопределяем глобальную конфигурацию
    import config as config_module
    config_module.CONFIG = config

    # Выводим информацию о конфигурации
    print(f"\n{'='*60}")
    print("НАСТРОЙКА RAG БОТА")
    print(f"{'='*60}")
    print(f"Корневая директория: {PROJECT_ROOT}")
    print(f"Векторная база: {config.vector_store.persist_directory}")
    print(f"  Коллекция: {config.vector_store.collection_name}")
    print(f"  Поиск: k={config.vector_store.search_kwargs.get('k')}, "
          f"threshold={config.vector_store.search_kwargs.get('score_threshold')}")
    print(f"Модель LLM: {config.llm.model_name}")
    print(f"Few-shot файл: {config.few_shot.examples_file}")
    print(f"  Используется примеров: {config.few_shot.examples_to_use}")

    # Получаем статистику few-shot примеров
    few_shot_stats = config.few_shot.get_statistics()
    print(f"  Загружено примеров: {few_shot_stats.get('total', 0)}")
    if few_shot_stats.get('using_fallback'):
        print(f"  ⚠️  Используются fallback примеры")

    print(f"Логирование: {config.logging.level}, файл: {config.logging.file_path}")
    print(f"{'='*60}\n")

    if args.mode == 'telegram':
        if not args.telegram_token:
            print("Ошибка: для режима telegram требуется указать --telegram-token")
            sys.exit(1)

        # Запускаем Telegram бота
        sys.argv = [sys.argv[0], '--token', args.telegram_token]
        run_telegram_bot()

    elif args.mode == 'test':
        # Запускаем тестирование
        from test_rag import main as run_test

        # Обновляем аргументы для тестирования
        test_args = [
            sys.argv[0],
            '--test-file', str(config.few_shot.examples_file),
            '--max-questions', str(args.max_questions),
            '--output', str(args.output)
        ]

        if args.categories:
            test_args.extend(['--categories'] + args.categories)

        sys.argv = test_args
        run_test()

    else:
        # Запускаем REPL интерфейс
        run_repl()


if __name__ == "__main__":
    main()