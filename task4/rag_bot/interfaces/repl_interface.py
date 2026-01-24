# interfaces/repl_interface.py
"""
Консольный REPL интерфейс для RAG-бота
"""

import sys
import json
from typing import Optional
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта
sys.path.append(str(Path(__file__).parent.parent))

from core.rag_pipeline import RAGPipeline
from config import CONFIG


class REPLInterface:
    """Интерфейс командной строки для RAG-бота"""

    def __init__(self, pipeline: Optional[RAGPipeline] = None):
        """
        Инициализация REPL интерфейса

        Args:
            pipeline: Экземпляр RAG пайплайна
        """
        self.pipeline = pipeline or RAGPipeline(CONFIG)
        self._print_welcome()

    def _print_welcome(self):
        """Вывод приветственного сообщения"""
        print("\n" + "="*60)
        print("RAG БОТ - Система вопросов и ответов на основе знаний")
        print("="*60)
        print("Команды:")
        print("  /help - Показать это сообщение")
        print("  /info - Информация о системе")
        print("  /exit или /quit - Выход")
        print("  /debug - Подробная информация о последнем ответе")
        print("="*60 + "\n")

    def _print_help(self):
        """Вывод справки"""
        print("\nДоступные команды:")
        print("  /help - Показать справку")
        print("  /info - Информация о системе")
        print("  /exit, /quit - Выход из программы")
        print("  /debug - Показать детали последнего ответа")
        print("\nПросто введите свой вопрос и нажмите Enter для получения ответа.\n")

    def _print_system_info(self):
        """Вывод информации о системе"""
        info = self.pipeline.get_pipeline_info()

        print("\n" + "="*60)
        print("ИНФОРМАЦИЯ О СИСТЕМЕ")
        print("="*60)
        print(f"Модель LLM: {info['llm_model']}")
        print(f"Модель эмбеддингов: {info['embedding_model']}")
        print(f"Few-shot примеров: {info['few_shot_examples']}")

        if "error" not in info['vector_store']:
            print(f"Векторная база: {info['vector_store']['name']}")
            print(f"Документов в базе: {info['vector_store']['count']}")
        else:
            print("Векторная база: Не доступна")
        print("="*60 + "\n")

    def _print_debug_info(self, response):
        """Вывод отладочной информации"""
        print("\n" + "="*60)
        print("ДЕТАЛИ ОТВЕТА")
        print("="*60)
        print(f"Время обработки: {response.processing_time:.2f} сек")
        print(f"Найдено документов: {len(response.retrieved_docs)}")
        print(f"Использовано источников: {len(response.used_sources)}")

        if response.used_sources:
            print(f"Источники: {', '.join(response.used_sources)}")

        if response.retrieved_docs:
            print("\nНайденные документы:")
            for i, doc in enumerate(response.retrieved_docs, 1):
                print(f"\n{i}. Схожесть: {doc['score']:.3f}")
                if 'source' in doc.get('metadata', {}):
                    print(f"   Источник: {doc['metadata']['source']}")
                preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                print(f"   Содержание: {preview}")
        print("="*60 + "\n")

    def run(self):
        """Запуск REPL интерфейса"""
        last_response = None

        while True:
            try:
                # Чтение ввода пользователя
                user_input = input("\nВопрос: ").strip()

                # Проверка команд
                if not user_input:
                    continue

                if user_input.lower() in ['/exit', '/quit']:
                    print("Выход из программы. До свидания!")
                    break

                elif user_input.lower() == '/help':
                    self._print_help()
                    continue

                elif user_input.lower() == '/info':
                    self._print_system_info()
                    continue

                elif user_input.lower() == '/debug':
                    if last_response:
                        self._print_debug_info(last_response)
                    else:
                        print("Нет информации о последнем ответе.")
                    continue

                # Обработка вопроса
                print("\nОбработка вопроса...")
                response = self.pipeline.process_query(user_input)
                last_response = response

                # Вывод ответа
                print(f"\nОтвет: {response.answer}")
                print(f"(Время обработки: {response.processing_time:.2f} сек)")

                if not response.has_context:
                    print("\n⚠️  Ответ основан на общих знаниях модели, а не на базе знаний.")

            except KeyboardInterrupt:
                print("\n\nВыход из программы. До свидания!")
                break
            except Exception as e:
                print(f"\nОшибка: {str(e)}")


def main():
    """Точка входа для REPL интерфейса"""
    try:
        repl = REPLInterface()
        repl.run()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()