# config.py
"""
Конфигурация RAG-бота с структурированными путями и настройками
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

# ============================================================================
# БАЗОВЫЕ ПУТИ И НАСТРОЙКИ
# ============================================================================

# Базовые пути
PROJECT_ROOT = Path(__file__).parent.parent.parent
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
GENERATED_PATH = PROJECT_ROOT / "generated"
LOGS_PATH = PROJECT_ROOT / "logs"
MODELS_CACHE_PATH = PROJECT_ROOT / "models_cache"

# Создаем необходимые директории
for path in [LOGS_PATH, MODELS_CACHE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Модель эмбеддингов
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384  # Размерность эмбеддингов для all-MiniLM-L6-v2

# Настройки поиска - УЛУЧШЕННЫЕ
DEFAULT_SEARCH_K = 15  # Больше результатов для лучшего контекста
DEFAULT_SCORE_THRESHOLD = 0.2  # Ниже порог для включения большего контекста
MAX_CONTEXT_LENGTH = 3000  # Максимальная длина контекста в токенах
MIN_RELEVANT_DOCS = 1  # Минимальное количество релевантных документов

# Few-shot файлы
FEW_SHOT_FILE_PATH = GENERATED_PATH / "few_shot_qa_pairs.jsonl"
FEW_SHOT_BACKUP_PATH = PROJECT_ROOT / "config" / "default_few_shot.jsonl"

# Настройки производительности
ENABLE_CACHE = True
CACHE_TTL_SECONDS = 3600  # Время жизни кэша
BATCH_SIZE = 32  # Размер батча для обработки
MAX_CONCURRENT_REQUESTS = 5  # Максимальное количество параллельных запросов

# Настройки логирования
LOG_LEVEL = "DEBUG"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_PATH / "rag_bot.log"

# Настройки ответов
MAX_ANSWER_LENGTH = 2000  # Максимальная длина ответа в символах
ENABLE_SOURCE_CITATION = True  # Включить цитирование источников
DEFAULT_LANGUAGE = "en"  # Язык ответов по умолчанию

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def load_few_shot_examples(file_path: Path) -> List[Dict[str, Any]]:
    """
    Загрузка few-shot примеров из JSONL файла

    Args:
        file_path: Путь к JSONL файлу

    Returns:
        List[Dict]: Список few-shot примеров

    Raises:
        FileNotFoundError: Если файл не найден
        json.JSONDecodeError: Если файл содержит невалидный JSON
    """
    # Пробуем основной файл
    if not file_path.exists():
        logger.warning(f"Основной файл few-shot примеров не найден: {file_path}")

        # Пробуем backup файл
        if FEW_SHOT_BACKUP_PATH.exists():
            logger.info(f"Используем backup файл: {FEW_SHOT_BACKUP_PATH}")
            file_path = FEW_SHOT_BACKUP_PATH
        else:
            logger.error("Backup файл также не найден")
            return []

    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)

                # Проверяем обязательные поля
                if not all(key in example for key in ['question', 'answer']):
                    logger.warning(f"Пропускаем пример в строке {line_num}: отсутствуют обязательные поля")
                    continue

                # Форматируем пример
                formatted_example = {
                    'question': example['question'],
                    'answer': example['answer'],
                    'source_docs': example.get('source_docs', []),
                    'difficulty': example.get('difficulty', 'unknown'),
                    'category': example.get('category', 'unknown'),
                    'context_required': example.get('context_required', True),
                    'reasoning_steps': example.get('reasoning_steps', []),
                    'metadata': example.get('metadata', {})
                }

                examples.append(formatted_example)

            except json.JSONDecodeError as e:
                logger.error(f"Ошибка JSON в строке {line_num}: {e}")
                continue
            except Exception as e:
                logger.error(f"Ошибка обработки строки {line_num}: {e}")
                continue

    logger.info(f"Загружено {len(examples)} few-shot примеров из {file_path}")
    return examples


def select_best_examples(
        examples: List[Dict[str, Any]],
        count: int = 2,
        difficulty_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        prioritize_with_context: bool = True
) -> List[Dict[str, Any]]:
    """
    Выбор лучших few-shot примеров для использования в промпте

    Args:
        examples: Все загруженные примеры
        count: Количество примеров для выбора
        difficulty_filter: Фильтр по сложности
        category_filter: Фильтр по категории
        prioritize_with_context: Приоритезировать примеры с context_required=True

    Returns:
        List[Dict]: Отобранные примеры
    """
    if not examples:
        return []

    filtered_examples = examples

    # Применяем фильтры, если указаны
    if difficulty_filter:
        filtered_examples = [
            ex for ex in filtered_examples
            if ex.get('difficulty', '').lower() == difficulty_filter.lower()
        ]
        logger.debug(f"После фильтра по сложности '{difficulty_filter}': {len(filtered_examples)} примеров")

    if category_filter:
        filtered_examples = [
            ex for ex in filtered_examples
            if ex.get('category', '').lower() == category_filter.lower()
        ]
        logger.debug(f"После фильтра по категории '{category_filter}': {len(filtered_examples)} примеров")

    # Если после фильтрации не осталось примеров, используем все
    if not filtered_examples:
        logger.warning("После фильтрации не осталось примеров, используем все доступные")
        filtered_examples = examples

    # Приоритезация примеров с context_required=True
    if prioritize_with_context:
        context_examples = [ex for ex in filtered_examples if ex.get('context_required', True)]
        no_context_examples = [ex for ex in filtered_examples if not ex.get('context_required', True)]

        # Если есть примеры с контекстом, используем их
        if context_examples:
            filtered_examples = context_examples
            logger.debug(f"Используем {len(filtered_examples)} примеров с контекстом")
        elif no_context_examples:
            filtered_examples = no_context_examples
            logger.debug(f"Используем {len(filtered_examples)} примеров без контекста")

    # Сортируем по сложности (легкие сначала) и ограничиваем количество
    difficulty_order = {'easy': 0, 'medium': 1, 'hard': 2, 'unknown': 3}

    sorted_examples = sorted(
        filtered_examples,
        key=lambda x: difficulty_order.get(x.get('difficulty', 'unknown'), 3)
    )

    selected = sorted_examples[:count]
    logger.info(f"Отобрано {len(selected)} few-shot примеров для использования")

    return selected


def get_default_fallback_examples() -> List[Dict[str, Any]]:
    """
    Получение fallback примеров (используются при отсутствии файла)

    Returns:
        List[Dict]: Fallback примеры
    """
    return [
        {
            "question": "What are the main ingredients of Obsidian Stone?",
            "answer": "The primary ingredient of Obsidian Stone is moonleaf harvested during specific celestial alignments, along with secondary components like star-moss and crystalized dew.",
            "source_docs": ["ENCY_001", "ENCY_002"],
            "difficulty": "easy",
            "category": "magic_items",
            "context_required": True,
            "metadata": {"is_fallback": True}
        },
        {
            "question": "What is Syloos's primary responsibility?",
            "answer": "Syloos is responsible for brewing the Obsidian Stone, studying medicinal herbs, preserving ancient knowledge, and serving as a healer for the community.",
            "source_docs": ["ENCY_001", "JOUR_001"],
            "difficulty": "easy",
            "category": "roles",
            "context_required": True,
            "metadata": {"is_fallback": True}
        }
    ]


# ============================================================================
# КЛАССЫ КОНФИГУРАЦИИ
# ============================================================================

@dataclass
class VectorStoreConfig:
    """Конфигурация векторного хранилища"""
    persist_directory: Path = CHROMA_DB_PATH
    collection_name: str = "knowledge_base"
    embedding_model: str = EMBEDDING_MODEL
    embedding_dimensions: int = EMBEDDING_DIMENSIONS
    search_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.search_kwargs is None:
            self.search_kwargs = {
                "k": DEFAULT_SEARCH_K,
                "score_threshold": DEFAULT_SCORE_THRESHOLD
            }

        # Проверяем существование директории
        if not self.persist_directory.exists():
            logger.warning(f"Директория векторной базы не найдена: {self.persist_directory}")
            logger.info("При первом запуске будет создана новая база")

@dataclass
class LLMConfig:
    """Конфигурация языковой модели"""
    model_name: str = "microsoft/Phi-3-mini-128k-instruct"
    model_cache_dir: Path = MODELS_CACHE_PATH
    temperature: float = 0.1
    max_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    device: str = "auto"  # ← auto будет определять DirectML/CPU
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    use_flash_attention: bool = False  # DirectML пока не поддерживает flash attention
    hf_token: Optional[str] = None
    enable_progress_bars: bool = False
    use_directml_optimizations: bool = True  # ← Новая опция для DirectML


@dataclass
class FewShotConfig:
    """Конфигурация Few-Shot примеров"""
    examples_file: Path = FEW_SHOT_FILE_PATH
    examples_to_use: int = 2
    difficulty_filter: Optional[str] = None  # "easy", "medium", "hard"
    category_filter: Optional[str] = None  # "magic_items", "characters", и т.д.
    prioritize_with_context: bool = True
    enable_fallback: bool = True
    _loaded_examples: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _selected_examples: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def load_examples(self) -> List[Dict[str, Any]]:
        """Загрузка примеров из файла"""
        if not self._loaded_examples:
            try:
                self._loaded_examples = load_few_shot_examples(self.examples_file)

                # Если файл не найден и включен fallback
                if not self._loaded_examples and self.enable_fallback:
                    logger.warning("Используются fallback few-shot примеры")
                    self._loaded_examples = get_default_fallback_examples()

            except Exception as e:
                logger.error(f"Ошибка загрузки few-shot примеров: {e}")

                if self.enable_fallback:
                    logger.warning("Используются fallback few-shot примеры из-за ошибки")
                    self._loaded_examples = get_default_fallback_examples()
                else:
                    self._loaded_examples = []

        return self._loaded_examples

    def get_selected_examples(self) -> List[Dict[str, Any]]:
        """Получение отобранных примеров для использования"""
        if not self._selected_examples:
            all_examples = self.load_examples()
            self._selected_examples = select_best_examples(
                all_examples,
                count=self.examples_to_use,
                difficulty_filter=self.difficulty_filter,
                category_filter=self.category_filter,
                prioritize_with_context=self.prioritize_with_context
            )

        return self._selected_examples

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики по примерам"""
        examples = self.load_examples()

        if not examples:
            return {"total": 0, "categories": {}, "difficulties": {}}

        # Статистика по категориям и сложности
        categories = {}
        difficulties = {}

        for example in examples:
            category = example.get('category', 'unknown')
            difficulty = example.get('difficulty', 'unknown')

            categories[category] = categories.get(category, 0) + 1
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        return {
            "total": len(examples),
            "loaded_from": str(self.examples_file),
            "using_fallback": any(ex.get('metadata', {}).get('is_fallback', False) for ex in examples),
            "categories": categories,
            "difficulties": difficulties,
            "selected_count": len(self.get_selected_examples())
        }


@dataclass
class PromptConfig:
    """Конфигурация промптов"""
    max_context_length: int = MAX_CONTEXT_LENGTH
    enable_source_citation: bool = ENABLE_SOURCE_CITATION
    default_language: str = DEFAULT_LANGUAGE
    answer_max_length: int = MAX_ANSWER_LENGTH
    include_confidence: bool = True  # Включать оценку уверенности в ответе

    # Шаблоны промптов
    system_prompt_template: str = """Ты - полезный ассистент, отвечающий на вопросы на основе предоставленного контекста.
Твоя задача - давать точные, информативные и полезные ответы, используя только информацию из контекста.
Если в контексте нет информации для ответа, честно скажи об этом."""

    few_shot_intro_template: str = "Вот несколько примеров правильных ответов на основе контекста:\n\n"

    context_template: str = """КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

ВАЖНЫЕ ИНСТРУКЦИИ:
1. Отвечай ТОЛЬКО на основе предоставленного контекста
2. Если информации в контексте недостаточно, скажи "На основе имеющейся информации не могу дать точный ответ"
3. Будь точным и конкретным
4. Указывай источники информации в формате [Документ X]
5. Если используешь информацию из нескольких источников, объедини её логично

ВОПРОС: {question}

ОТВЕТ: """


@dataclass
class PerformanceConfig:
    """Конфигурация производительности"""
    enable_cache: bool = ENABLE_CACHE
    cache_ttl_seconds: int = CACHE_TTL_SECONDS
    batch_size: int = BATCH_SIZE
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS
    max_processing_time: int = 30  # Максимальное время обработки в секундах

    def __post_init__(self):
        # Проверка значений
        if self.max_processing_time < 5:
            logger.warning(f"max_processing_time слишком низкое: {self.max_processing_time}")


@dataclass
class LoggingConfig:
    """Конфигурация логирования"""
    level: str = LOG_LEVEL
    format: str = LOG_FORMAT
    file_path: Path = LOG_FILE
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5  # Количество backup файлов

    def setup_logging(self):
        """Настройка логирования"""
        import logging.handlers

        # Форматтер
        formatter = logging.Formatter(self.format)

        # Обработчик для файла
        file_handler = logging.handlers.RotatingFileHandler(
            self.file_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)

        # Обработчик для консоли
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Настройка корневого логгера
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.level))

        # Очищаем существующие обработчики
        root_logger.handlers.clear()

        # Добавляем обработчики
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        logger.info(f"Логирование настроено. Уровень: {self.level}, Файл: {self.file_path}")


@dataclass
class RAGConfig:
    """Общая конфигурация RAG"""
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    few_shot: FewShotConfig = field(default_factory=FewShotConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    project_root: Path = field(default=PROJECT_ROOT, init=False)

    def __post_init__(self):
        # Настраиваем логирование
        self.logging.setup_logging()

        # Логируем информацию о конфигурации
        logger.info("=" * 60)
        logger.info("ИНИЦИАЛИЗАЦИЯ RAG КОНФИГУРАЦИИ")
        logger.info("=" * 60)
        logger.info(f"Корневая директория: {self.project_root}")
        logger.info(f"Векторная база: {self.vector_store.persist_directory}")
        logger.info(f"Модель LLM: {self.llm.model_name}")
        logger.info(f"Few-shot файл: {self.few_shot.examples_file}")
        logger.info("=" * 60)


# ============================================================================
# ФАБРИКА КОНФИГУРАЦИИ
# ============================================================================

def create_config(
        vector_store_dir: Optional[Union[str, Path]] = None,
        collection_name: Optional[str] = None,
        model_name: Optional[str] = None,
        few_shot_file: Optional[Union[str, Path]] = None,
        examples_to_use: Optional[int] = None,
        search_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        log_level: Optional[str] = None
) -> RAGConfig:
    """
    Фабрика для создания конфигурации

    Args:
        vector_store_dir: Директория с векторной базой
        collection_name: Название коллекции
        model_name: Название модели LLM
        few_shot_file: Путь к файлу с few-shot примерами
        examples_to_use: Количество примеров для использования
        search_k: Количество документов для поиска
        score_threshold: Порог схожести
        log_level: Уровень логирования

    Returns:
        RAGConfig: Сконфигурированный объект
    """
    # Создаем базовую конфигурацию
    config = RAGConfig()

    # Обновляем параметры векторного хранилища
    if vector_store_dir:
        config.vector_store.persist_directory = Path(vector_store_dir)

    if collection_name:
        config.vector_store.collection_name = collection_name

    if search_k is not None:
        config.vector_store.search_kwargs["k"] = search_k

    if score_threshold is not None:
        config.vector_store.search_kwargs["score_threshold"] = score_threshold

    # Обновляем параметры LLM
    if model_name:
        config.llm.model_name = model_name

    # Обновляем параметры few-shot
    if few_shot_file:
        config.few_shot.examples_file = Path(few_shot_file)

    if examples_to_use:
        config.few_shot.examples_to_use = examples_to_use

    # Обновляем логирование
    if log_level:
        config.logging.level = log_level

    return config


# ============================================================================
# ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ
# ============================================================================

# Глобальный экземпляр конфигурации
CONFIG = create_config()

# Экспортируем часто используемые пути и настройки
__all__ = [
    'CONFIG',
    'PROJECT_ROOT',
    'KNOWLEDGE_BASE_PATH',
    'CHROMA_DB_PATH',
    'GENERATED_PATH',
    'LOGS_PATH',
    'MODELS_CACHE_PATH',
    'EMBEDDING_MODEL',
    'EMBEDDING_DIMENSIONS',
    'DEFAULT_SEARCH_K',
    'DEFAULT_SCORE_THRESHOLD',
    'FEW_SHOT_FILE_PATH',
    'create_config',
    'load_few_shot_examples',
    'select_best_examples'
]