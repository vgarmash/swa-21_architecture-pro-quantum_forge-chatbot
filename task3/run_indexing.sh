#!/bin/bash
# run_indexing.sh - скрипт для создания индекса

set -e  # Выход при ошибке

echo "=== Задание 3: Создание векторного индекса ==="
echo ""

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден!"
    exit 1
fi

echo "🐍 Python версия: $(python3 --version)"

# Проверка папки с документами
if [ ! -d "../knowledge_base" ]; then
    echo "❌ Папка ../knowledge_base не найдена!"
    echo "   Создайте папку и добавьте документы"
    exit 1
fi

echo "📁 Документы: ../knowledge_base"

# Переходим в папку скрипта
cd "$(dirname "$0")"
echo "📁 Рабочая папка: $(pwd)"

# Проверяем requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "❌ Файл requirements.txt не найден!"
    exit 1
fi

echo "📦 Установка зависимостей..."

# Устанавливаем/обновляем pip
python3 -m pip install --upgrade pip

# Устанавливаем зависимости
pip install -r requirements.txt

echo ""
echo "✅ Зависимости установлены"
echo ""

# Проверяем версии
echo "📊 Установленные версии:"
python3 -c "
try:
    import langchain
    print(f'   langchain: {langchain.__version__}')
except ImportError:
    print('   ❌ langchain не установлен')

try:
    import langchain_text_splitters
    print(f'   langchain_text_splitters: установлен')
except ImportError:
    print('   ❌ langchain_text_splitters не установлен')

try:
    import chromadb
    print(f'   chromadb: {chromadb.__version__}')
except ImportError:
    print('   ❌ chromadb не установлен')

try:
    import sentence_transformers
    print(f'   sentence-transformers: установлен')
except ImportError:
    print('   ❌ sentence-transformers не установлен')
"

echo ""
echo "🚀 Запуск создания индекса..."
echo "=" * 50

# Запускаем скрипт
python3 build_index.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=" * 50
    echo "✅ ИНДЕКС УСПЕШНО СОЗДАН!"
    echo "=" * 50
    echo ""
    echo "📊 Для поиска запустите:"
    echo "   python3 query_index.py --mode interactive"
    echo ""
    echo "📁 Индекс сохранен в: $(pwd)/chroma_db"
else
    echo ""
    echo "❌ ОШИБКА ПРИ СОЗДАНИИ ИНДЕКСА"
    exit 1
fi