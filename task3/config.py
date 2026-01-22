"""
config.py - Конфигурационные параметры для индексации
"""

from pathlib import Path

# Базовые пути
PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# Параметры индексации
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Модель эмбеддингов
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# Настройки поиска - УЛУЧШЕННЫЕ
DEFAULT_SEARCH_K = 15  # Больше результатов
DEFAULT_SCORE_THRESHOLD = 0.2  # Ниже порог

# Поддерживаемые форматы
SUPPORTED_EXTENSIONS = {
    '.txt': 'text',
    '.md': 'markdown',
    '.pdf': 'pdf',
}