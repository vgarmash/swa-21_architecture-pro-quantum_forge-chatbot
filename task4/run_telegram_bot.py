import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from rag_engine import get_rag_chain

# Вставьте сюда токен вашего бота
TELEGRAM_TOKEN = "ВАШ_ТОКЕН_ОТ_BOTFATHER"

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Глобальная переменная для цепочки (чтобы не грузить её при каждом сообщении)
rag_chain = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! I am a RAG bot about the fantasy world. Ask me anything!')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global rag_chain
    query = update.message.text

    # Эмуляция "печатает..."
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        # Генерация ответа
        response = rag_chain.invoke(query)
        await update.message.reply_text(response)
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        await update.message.reply_text("Sorry, I encountered an error processing your request.")

if __name__ == '__main__':
    print("[Telegram Bot] Initializing RAG engine...")
    try:
        rag_chain = get_rag_chain()
    except Exception as e:
        print(f"Failed to initialize RAG engine: {e}")
        exit(1)

    print("[Telegram Bot] Starting polling...")
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    start_handler = CommandHandler('start', start)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)

    application.add_handler(start_handler)
    application.add_handler(message_handler)

    application.run_polling()