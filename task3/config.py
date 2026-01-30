"""
config.py - Конфигурационные параметры для индексирования базы знаний
"""

from pathlib import Path

# Базовые пути
PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# Параметры векторного хранилища
COLLECTION_NAME = "knowledge_base"

# Модель эмбеддингов
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# Настройки чанков (в токенах)
TARGET_CHUNK_TOKENS = 400
MIN_CHUNK_TOKENS = 200
MAX_CHUNK_TOKENS = 500
CHUNK_OVERLAP_RATIO = 0.15

# Дополнительные настройки поиска
DEFAULT_SEARCH_K = 15
DEFAULT_SCORE_THRESHOLD = 0.2