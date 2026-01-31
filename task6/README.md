# Task 6 — Обновление и автоматизация индекса Chroma DB

## Скрипт обновления индекса
- **Файл:** [`update_index.py`](update_index.py)
- **Назначение:** синхронизирует коллекцию Chroma `knowledge_base` с папкой [`knowledge_base/`](../knowledge_base).
- **Функции:**
  - Вычисляет SHA256 для каждого документа и ищет расхождения с метаданными векторов.
  - Добавляет новые/изменённые чанки, удаляет устаревшие.
  - Генерирует эмбеддинги через `sentence-transformers/all-MiniLM-L6-v2`.
  - Логирует процесс в [`update_index.log`](update_index.log).
  - Поддерживает fallback для разных версий `langchain_chroma` и `langchain_huggingface`.

### Запуск вручную
```bash
cd task6
python update_index.py
```

Логи записываются в [`update_index.log`](update_index.log).

## Планировщик обновлений (Windows)
- **Файл:** [`schedule_update.ps1`](schedule_update.ps1)
- **Назначение:** регистрирует и исполняет ежедневный запуск обновления на Windows Task Scheduler.
- **Параметры:**
  - `-Register` (по умолчанию) — регистрирует задачу `Task6_UpdateChromaIndex`, которая запускается каждый день в 06:00.
  - `-RunJob` — выполняет обновление сразу, с одной повторной попыткой через 30 секунд при ошибке, а результат логирует в [`update_index.log`](update_index.log).

### Регистрация задачи
```powershell
powershell -ExecutionPolicy Bypass -File .\schedule_update.ps1
```

### Ручной запуск (тест)
```powershell
powershell -ExecutionPolicy Bypass -File .\schedule_update.ps1 -RunJob
```

## Зависимости
- **Файл:** [`requirements.txt`](requirements.txt)
- Устанавливает минимально необходимый набор библиотек (`chromadb`, `langchain-*`, `sentence-transformers` и др.).

### Установка зависимостей
```bash
cd task6
python -m pip install -r requirements.txt
```

## Поведение при ошибках
- `update_index.py` — ошибка логируется и повторное выполнение не предпринимается автоматически.
- `schedule_update.ps1 -RunJob` — при неудаче делает одно повторение через 30 секунд; при двух ошибках подряд завершает работу с кодом ошибки.
