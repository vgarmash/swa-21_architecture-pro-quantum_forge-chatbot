# check_config.py
"""
Скрипт для проверки путей и конфигурации
"""

import sys
from pathlib import Path

# Добавляем родительскую директорию в путь
sys.path.append(str(Path(__file__).parent))

from config import (
    PROJECT_ROOT,
    KNOWLEDGE_BASE_PATH,
    CHROMA_DB_PATH,
    GENERATED_PATH,
    LOGS_PATH,
    MODELS_CACHE_PATH,
    FEW_SHOT_FILE_PATH,
    EMBEDDING_MODEL,
    DEFAULT_SEARCH_K,
    DEFAULT_SCORE_THRESHOLD,
    CONFIG
)


def check_paths():
    """Проверка существования путей"""
    print("=" * 60)
    print("ПРОВЕРКА ПУТЕЙ КОНФИГУРАЦИИ")
    print("=" * 60)

    paths_to_check = [
        ("Корневая директория проекта", PROJECT_ROOT),
        ("Директория базы знаний", KNOWLEDGE_BASE_PATH),
        ("Директория векторной базы", CHROMA_DB_PATH),
        ("Директория generated", GENERATED_PATH),
        ("Директория логов", LOGS_PATH),
        ("Кэш моделей", MODELS_CACHE_PATH),
        ("Файл few-shot примеров", FEW_SHOT_FILE_PATH),
    ]

    for name, path in paths_to_check:
        exists = path.exists()
        print(f"{name:30} {path}")
        print(f"{'':30} {'✓ Существует' if exists else '✗ Отсутствует'}")

        if exists and path.is_file():
            try:
                size = path.stat().st_size
                print(f"{'':30} Размер: {size:,} байт")
            except:
                pass

        print()

    print("=" * 60)


def check_config_values():
    """Проверка значений конфигурации"""
    print("НАСТРОЙКИ КОНФИГУРАЦИИ")
    print("=" * 60)

    print(f"Модель эмбеддингов: {EMBEDDING_MODEL}")
    print(f"Размерность эмбеддингов: {CONFIG.vector_store.embedding_dimensions}")
    print(f"Поиск документов (k): {DEFAULT_SEARCH_K}")
    print(f"Порог схожести: {DEFAULT_SCORE_THRESHOLD}")

    # Информация о LLM
    print(f"\nМодель LLM: {CONFIG.llm.model_name}")
    print(f"Температура: {CONFIG.llm.temperature}")
    print(f"Максимальное количество токенов: {CONFIG.llm.max_tokens}")
    print(f"Устройство: {CONFIG.llm.device}")
    print(f"Кэш моделей: {CONFIG.llm.model_cache_dir}")

    # Информация о few-shot
    few_shot_stats = CONFIG.few_shot.get_statistics()
    print(f"\nFew-shot примеры:")
    print(f"  Файл: {CONFIG.few_shot.examples_file}")
    print(f"  Используется: {CONFIG.few_shot.examples_to_use}")
    print(f"  Загружено всего: {few_shot_stats.get('total', 0)}")

    if few_shot_stats.get('categories'):
        print(f"  Категории: {', '.join(few_shot_stats['categories'].keys())}")

    if few_shot_stats.get('using_fallback'):
        print(f"  ⚠️  Используются fallback примеры")

    print("=" * 60)


def check_environment():
    """Проверка окружения"""
    import platform
    import sys

    print("\nИНФОРМАЦИЯ О СРЕДЕ")
    print("=" * 60)
    print(f"ОС: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Архитектура: {platform.architecture()[0]}")
    print("=" * 60)


def main():
    """Основная функция проверки"""
    try:
        check_paths()
        check_config_values()
        check_environment()

        print("\n✓ Проверка конфигурации завершена успешно!")

    except Exception as e:
        print(f"\n✗ Ошибка при проверке конфигурации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()