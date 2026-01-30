import logging
import os
from typing import Optional

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from rag_engine import RagEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

_engine: Optional[RagEngine] = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "Привет! Я RAG-бот. Задайте вопрос по базе знаний, и я постараюсь помочь."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.text is None:
        return

    message_text = update.message.text.strip()
    if not message_text:
        await update.message.reply_text("Пожалуйста, отправьте непустой вопрос.")
        return

    if _engine is None:
        logging.error("RagEngine не инициализирован.")
        await update.message.reply_text(
            "Бот временно недоступен. Попробуйте позже."
        )
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    try:
        response = _engine.search(message_text)
        answer = response.get(
            "answer", "Information is unavailable. Ask another question."
        )
        await update.message.reply_text(answer)
    except Exception as exc:  # noqa: BLE001
        logging.exception("Ошибка при обработке сообщения: %s", exc)
        await update.message.reply_text(
            "Произошла ошибка при обработке вашего запроса."
        )


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logging.error(
            "Переменная окружения TELEGRAM_BOT_TOKEN не установлена. "
            "Задайте токен перед запуском бота."
        )
        raise SystemExit(1)

    global _engine
    logging.info("Инициализация RagEngine для Telegram-бота")
    _engine = RagEngine()

    application = ApplicationBuilder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    )

    logging.info("Запуск Telegram-бота")
    application.run_polling()


if __name__ == "__main__":
    main()
