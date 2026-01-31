# План восстановления поддержки токенизатора cl100k_base в chonkie

## Цель
Добиться корректной работы `chonkie.RecursiveChunker` и `OverlapRefinery` в `build_index.py` с использованием токенизатора `cl100k_base`, устранив предупреждения о невозможности загрузки токенизатора.

## Актуальное состояние
- Скрипт [`build_index.py`](task3/build_index.py) использует токенизатор `tiktoken/cl100k_base` через библиотеку `chonkie`.
- При запуске ранее возникала цепочка предупреждений: отсутствовали подходящие реализации токенизатора в `tokenizers`, затем `tiktoken`, затем `transformers`.
- В результате `RecursiveChunker` и `OverlapRefinery` не инициализировались, и система переключалась на fallback sentence-aware chunker.

## Выявленные пробелы в зависимостях
1. В `task3/requirements.txt` файл был закодирован в UTF-16 LE; пересохранен в UTF-8 с явным перечислением зависимостей.
2. Для работы `chonkie` с рецептом `markdown` требуется токенизатор, доступный через одну из библиотек: `tokenizers`, `tiktoken` или `transformers`.
3. Виртуальная среда уже содержит пакеты `tiktoken>=0.5.2` (установлена версия 0.12.0) и `tokenizers>=0.15.2` (установлена версия 0.22.2). Однако `chonkie` по-прежнему не обнаруживает `cl100k_base`.

## Обновление зависимостей
- Пересохранен [`task3/requirements.txt`](task3/requirements.txt) со списком:
  ```
  langchain==1.2.4
  chromadb==1.4.1
  langchain-chroma==1.1.0
  langchain-classic==1.0.1
  langchain-community==0.4.1
  langchain-core==1.2.7
  langchain-huggingface==1.2.0
  langchain-openai==1.1.7
  langchain-text-splitters==1.1.0
  openai==2.16.0
  sentence-transformers==5.2.0
  chonkie[hub]==1.5.4
  tiktoken>=0.5.2
  tokenizers>=0.15.2
  ```
- Команда `pip install -r task3/requirements.txt` подтверждает, что все зависимости удовлетворены (модуль `langchain_chroma` импортируется из пакета `langchain-chroma`).

## Текущий результат запуска
- Команда `python task3/build_index.py` завершается успешно, но `chonkie` всё ещё выводит предупреждения о невозможности загрузки токенизатора и использует fallback-чанкер.
- Логи подтверждают: `RecursiveChunker` не найден, `OverlapRefinery` не инициализирован.

## Дальнейшие шаги
1. Проверить, какой именно путь пытается открыть `chonkie` для токенизатора `cl100k_base` (возможно, ожидается ресурс `tokenizer.json`).
2. Исследовать документацию `chonkie` 1.5.4: требуется ли установка дополнительных файлов-моделей (например, `huggingface-hub` загрузка).
3. Вызвать `RecursiveChunker.available_tokenizers()` или аналогичную функцию, чтобы увидеть поддержку.
4. При необходимости указать токенизатор явно через объект `tiktoken.get_encoding("cl100k_base")` и передать в `RecursiveChunker` вместо строки.
5. Рассмотреть возможность обновления `chonkie` до последней версии (если >1.5.4) и проверить changelog на предмет исправлений загрузки токенизаторов.

## Критерии готовности
- Скрипт [`build_index.py`](task3/build_index.py) запускается без предупреждений о токенизаторах.
- В логах фиксируется использование `chonkie RecursiveChunker markdown recipe | tokenizer=tiktoken/cl100k_base` и успешная инициализация `OverlapRefinery`.
- Документация содержит актуальные инструкции по зависимостям и устранению проблем.
