
## Настройка доступа к OpenAI через lemonade-server

Для использования OpenAI API через lemonade-server необходимо выполнить следующие шаги:

1. Убедитесь, что lemonade-server запущен и доступен по адресу `http://localhost:8000/api/v1`
2. Установите необходимые зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Запустите REPL:
   ```bash
   python run_repl.py
   ```

Параметры по умолчанию:
- `OPENAI_BASE_URL` - URL сервера OpenAI (по умолчанию `http://localhost:8000/api/v1`)
- `OPENAI_API_KEY` - ключ API (по умолчанию `lemonade`)
- `OPENAI_MODEL` - модель LLM (по умолчанию `Mistral-7B-v0.3-Instruct-Hybrid`)

Если вы используете другой сервер или модель, измените соответствующие параметры в файле `llm_client.py`.