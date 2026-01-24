# quick_test.py
"""
Быстрый тест RAG с текущей конфигурацией из config.py
"""

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent))

from config import CONFIG
from core.rag_pipeline import RAGPipeline

# Проверяем DirectML
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except ImportError:
    DIRECTML_AVAILABLE = False
    torch_directml = None


def quick_test():
    """Быстрый тест с текущей конфигурацией из config.py"""

    print("=" * 70)
    print("БЫСТРЫЙ ТЕСТ RAG СИСТЕМЫ")
    print("=" * 70)

    # Показываем текущую конфигурацию
    print(f"Модель: {CONFIG.llm.model_name}")
    print(f"Устройство: {CONFIG.llm.device}")
    print(f"Max tokens: {CONFIG.llm.max_tokens}")
    print(f"Few-shot примеров: {CONFIG.few_shot.examples_to_use}")
    print(f"Поиск документов (k): {CONFIG.vector_store.search_kwargs.get('k')}")
    print(f"Порог схожести: {CONFIG.vector_store.search_kwargs.get('score_threshold')}")

    # Информация о DirectML
    if DIRECTML_AVAILABLE:
        if torch_directml.is_available():
            print(f"DirectML доступен: Да")
            print(f"DirectML устройств: {torch_directml.device_count()}")
            if torch_directml.device_count() > 0:
                print(f"DirectML устройство 0: {torch_directml.device_name(0)}")
        else:
            print(f"DirectML доступен: Нет (torch-directml установлен, но устройства не найдены)")
    else:
        print(f"DirectML доступен: Нет (torch-directml не установлен)")

    # Получаем статистику по few-shot примерам
    few_shot_stats = CONFIG.few_shot.get_statistics()
    print(f"Few-shot загружено: {few_shot_stats.get('total', 0)} примеров")

    print("-" * 70)

    # Создаем пайплайн
    print("Инициализация RAG пайплайна...")
    start_init = time.time()

    try:
        pipeline = RAGPipeline(CONFIG)
        init_time = time.time() - start_init
        print(f"✓ Пайплайн инициализирован за {init_time:.1f} сек")
    except Exception as e:
        print(f"✗ Ошибка инициализации: {e}")
        import traceback
        traceback.print_exc()
        return

    # Тестовые вопросы (можно адаптировать под предметную область)
    test_questions = [
        "What are the main ingredients of Obsidian Stone?",
        "Who is Quenix?",
        "What does Syloos do?",
        "How are standing stones classified?"
    ]

    print(f"\nТестирование {len(test_questions)} вопросов...")
    print("-" * 70)

    total_time = 0
    successful = 0
    with_context = 0

    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Вопрос: {question}")

        start = time.time()

        try:
            # Используем параметры из конфигурации
            response = pipeline.process_query(
                question,
                k=CONFIG.vector_store.search_kwargs.get('k', 5),
                similarity_threshold=CONFIG.vector_store.search_kwargs.get('score_threshold', 0.2)
            )

            elapsed = time.time() - start
            total_time += elapsed

            if response.answer and not response.answer.startswith("Ошибка"):
                successful += 1
                status = "✓"
            else:
                status = "✗"

            if response.has_context:
                with_context += 1

            print(f"   {status} Время: {elapsed:.1f} сек")
            print(f"   Контекст: {'есть' if response.has_context else 'нет'}")
            print(f"   Документов: {len(response.retrieved_docs)}")

            # Показываем часть ответа
            if response.answer:
                answer_preview = response.answer[:100] + "..." if len(response.answer) > 100 else response.answer
                print(f"   Ответ: {answer_preview}")
            else:
                print(f"   Ответ: нет ответа")

            if elapsed > 30:
                print(f"   ⚠️  Долго (> 30 сек)")
            elif elapsed > 10:
                print(f"   ⚠️  Приемлемо")
            else:
                print(f"   ✅ Быстро")

        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            print(f"   ✗ Ошибка за {elapsed:.1f} сек: {e}")
            import traceback
            traceback.print_exc()

    # Статистика
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ТЕСТА")
    print("=" * 70)

    avg_time = total_time / len(test_questions) if test_questions else 0
    success_rate = (successful / len(test_questions)) * 100 if test_questions else 0
    context_rate = (with_context / len(test_questions)) * 100 if test_questions else 0

    print(f"Всего вопросов: {len(test_questions)}")
    print(f"Успешно обработано: {successful} ({success_rate:.1f}%)")
    print(f"С контекстом: {with_context} ({context_rate:.1f}%)")
    print(f"Среднее время ответа: {avg_time:.1f} сек")
    print(f"Общее время теста: {total_time:.1f} сек")

    # Оценка производительности
    print("\nОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ:")
    if avg_time < 3:
        print("✅ Отличная производительность! Можно использовать в интерактивном режиме.")
    elif avg_time < 10:
        print("⚠️  Приемлемая производительность. Подходит для неинтерактивного использования.")
    elif avg_time < 30:
        print("⚠️  Медленно. Рассмотрите использование меньшей модели или оптимизаций.")
    else:
        print("❌ Очень медленно. Необходимо сменить модель или добавить GPU.")

    # Рекомендации
    print("\nРЕКОМЕНДАЦИИ:")

    if "phi-3" in CONFIG.llm.model_name.lower():
        model_size = "mini" if "mini" in CONFIG.llm.model_name.lower() else "small"
        print(f"1. Используется Phi-3-{model_size}, хороший выбор для CPU/DirectML")
    elif "qwen" in CONFIG.llm.model_name.lower():
        print("1. Qwen модель, возможно стоит перейти на Phi-3 для CPU/DirectML")
    else:
        print("1. Проверьте что модель оптимизирована для CPU/DirectML")

    if CONFIG.llm.max_tokens > 512:
        print(f"2. Max tokens ({CONFIG.llm.max_tokens}) высокий, можно уменьшить до 256 для ускорения")

    if CONFIG.vector_store.search_kwargs.get('k', 5) > 5:
        print(f"3. Поиск документов (k={CONFIG.vector_store.search_kwargs.get('k', 5)}) высокий, можно уменьшить")

    if DIRECTML_AVAILABLE and torch_directml.is_available():
        print("4. Используется DirectML (AMD GPU). Для лучшей производительности:")
        print("   - Убедитесь что драйвера AMD обновлены")
        print("   - Используйте float16 (автоматически включено)")
        print("   - Уменьшите max_tokens если не хватает VRAM")
    elif CONFIG.llm.device == "cpu":
        print("4. Используется CPU. Для ускорения:")
        print("   - Установите torch-directml для AMD GPU")
        print("   - Используйте меньшую модель")
        print("   - Уменьшите max_tokens и k поиска")

    if avg_time > 10:
        print("\n5. Рассмотрите использование команды для тестирования моделей:")
        print("   python test_phi3.py")

    print("=" * 70)


if __name__ == "__main__":
    quick_test()