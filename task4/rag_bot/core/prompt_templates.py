# core/prompt_templates.py
"""
Шаблоны промптов для RAG с техникой Few-Shot
"""

from typing import List, Dict, Any
import logging
from config import CONFIG

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Строитель промптов для RAG"""

    def __init__(self, config=None):
        """
        Инициализация строителя промптов

        Args:
            config: Конфигурация RAG
        """
        self.config = config or CONFIG
        self.prompt_config = self.config.prompt
        self.few_shot_config = self.config.few_shot
        self._few_shot_examples = None

    @property
    def few_shot_examples(self) -> List[Dict[str, Any]]:
        """Получение few-shot примеров (ленивая загрузка)"""
        if self._few_shot_examples is None:
            self._few_shot_examples = self.few_shot_config.get_selected_examples()
            logger.info(f"Используется {len(self._few_shot_examples)} few-shot примеров")
        return self._few_shot_examples

    def build_few_shot_section(self) -> str:
        """
        Построение секции с Few-Shot примерами

        Returns:
            str: Текст с примерами
        """
        examples = self.few_shot_examples

        if not examples:
            return ""

        examples_text_parts = [self.prompt_config.few_shot_intro_template]

        for i, example in enumerate(examples, 1):
            question = example['question']
            answer = example['answer']
            category = example.get('category', 'unknown')
            difficulty = example.get('difficulty', 'unknown')

            examples_text_parts.append(f"Пример {i} (Категория: {category}, Сложность: {difficulty}):")
            examples_text_parts.append(f"Вопрос: {question}")
            examples_text_parts.append(f"Ответ: {answer}")

            # Добавляем источники, если они есть и включено цитирование
            if self.prompt_config.enable_source_citation:
                source_docs = example.get('source_docs', [])
                if source_docs:
                    examples_text_parts.append(f"Источники: {', '.join(source_docs)}")

            examples_text_parts.append("-" * 40)

        # Удаляем последний разделитель
        if examples_text_parts and examples_text_parts[-1].startswith("-"):
            examples_text_parts.pop()

        return "\n".join(examples_text_parts) + "\n\n"

    def build_context_section(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Построение секции с найденными документами

        Args:
            retrieved_docs: Найденные документы из векторной БД

        Returns:
            str: Форматированный контекст
        """
        if not retrieved_docs:
            return "Контекст: Информация не найдена в базе знаний.\n"

        context_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            content = doc['content'].strip()
            metadata = doc.get('metadata', {})
            source = metadata.get('source', 'Неизвестный источник')
            doc_id = metadata.get('id', f'Док_{i}')
            score = doc.get('score', 0.0)

            # Форматируем информацию о документе
            doc_info = f"[Документ {i}"
            if doc_id:
                doc_info += f", ID: {doc_id}"
            if source:
                doc_info += f", Источник: {source}"
            doc_info += f", Схожесть: {score:.3f}]"

            context_parts.append(doc_info)

            # Обрезаем длинный контент для читаемости
            max_content_length = 800  # Увеличено для лучшего контекста
            if len(content) > max_content_length:
                # Пытаемся обрезать по предложению
                if '.' in content[:max_content_length]:
                    cut_point = content[:max_content_length].rfind('.') + 1
                    content = content[:cut_point] + " [текст продолжается...]"
                else:
                    content = content[:max_content_length] + "... [текст обрезан]"

            context_parts.append(f"{content}\n")

        # Объединяем контекст
        context_text = "\n".join(context_parts)

        # Проверяем длину контекста (примерная оценка)
        if len(context_text.split()) > self.prompt_config.max_context_length // 4:  # Примерная оценка
            logger.warning(f"Контекст может быть слишком длинным: {len(context_text.split())} слов")

        return context_text

    # core/prompt_templates.py - обновим build_rag_prompt
    def build_rag_prompt(
            self,
            query: str,
            retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Построение промпта в формате для модели (Phi-3 или общий)
        """
        # Строим секции промпта
        few_shot_section = self.build_few_shot_section()
        context_section = self.build_context_section(retrieved_docs)

        # Определяем формат для модели
        model_name = self.config.llm.model_name.lower()

        if "phi-3" in model_name:
            # Формат для Phi-3
            messages = [
                {"role": "system", "content": self.prompt_config.system_prompt_template},
                {"role": "user", "content": f"{few_shot_section}\n{context_section}\n\nВопрос: {query}\n\nОтветь на основе контекста:"}
            ]

            # Применяем чат-шаблон
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except:
                # Fallback к общему формату
                pass

        # Общий формат для других моделей
        prompt = f"""{few_shot_section}
    {context_section}
    На основе приведенного выше контекста ответь на вопрос.
    
    Вопрос: {query}
    
    Ответ: """

        return prompt

    def get_prompt_info(self) -> Dict[str, Any]:
        """
        Получение информации о промптах

        Returns:
            Dict: Информация о настройках промптов
        """
        return {
            "max_context_length": self.prompt_config.max_context_length,
            "enable_source_citation": self.prompt_config.enable_source_citation,
            "default_language": self.prompt_config.default_language,
            "answer_max_length": self.prompt_config.answer_max_length,
            "few_shot_examples_count": len(self.few_shot_examples)
        }