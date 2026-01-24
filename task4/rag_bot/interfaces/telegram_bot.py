# interfaces/telegram_bot.py
"""
Telegram бот для RAG системы (обновлено для python-telegram-bot 22.5)
"""

import asyncio
import logging
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path
import sys

# Добавляем родительскую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from telegram.error import TelegramError

from core.rag_pipeline import RAGPipeline
from config import CONFIG


# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Конфигурация Telegram бота"""
    token: str
    admin_ids: list = None
    rate_limit_per_user: int = 5  # запросов в минуту
    max_message_length: int = 4000  # максимальная длина сообщения


class RAGTelegramBot:
    """Telegram бот с RAG функциональностью"""

    def __init__(self, bot_config: BotConfig, rag_pipeline: RAGPipeline = None):
        """
        Инициализация Telegram бота

        Args:
            bot_config: Конфигурация бота
            rag_pipeline: Экземпляр RAG пайплайна
        """
        self.bot_config = bot_config
        self.pipeline = rag_pipeline or RAGPipeline(CONFIG)
        self.application = None
        self.user_requests = {}  # Для отслеживания запросов пользователей

        # Статистика
        self.stats = {
            "total_requests": 0,
            "successful_responses": 0,
            "errors": 0
        }

    def _check_rate_limit(self, user_id: int) -> bool:
        """
        Проверка ограничения запросов

        Args:
            user_id: ID пользователя

        Returns:
            bool: Превышен ли лимит
        """
        import time
        current_time = time.time()

        if user_id not in self.user_requests:
            self.user_requests[user_id] = []

        # Очищаем старые запросы (старше 60 секунд)
        self.user_requests[user_id] = [
            req_time for req_time in self.user_requests[user_id]
            if current_time - req_time < 60
        ]

        # Проверяем лимит
        if len(self.user_requests[user_id]) >= self.bot_config.rate_limit_per_user:
            return False

        # Добавляем текущий запрос
        self.user_requests[user_id].append(current_time)
        return True

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик команды /start

        Args:
            update: Объект обновления Telegram
            context: Контекст бота
        """
        user = update.effective_user
        welcome_text = f"""
Привет, {user.first_name}! 👋

Я - RAG бот, система вопросов и ответов на основе базы знаний.

📚 Я могу:
• Искать информацию в базе знаний
• Отвечать на вопросы на основе найденных данных
• Говорить "не знаю", если информации нет в базе

Доступные команды:
/start - Приветствие
/help - Помощь
/info - Информация о системе
/stats - Статистика бота

Просто задайте вопрос, и я постараюсь ответить на него!
        """
        await update.message.reply_text(welcome_text)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик команды /help

        Args:
            update: Объект обновления Telegram
            context: Контекст бота
        """
        help_text = """
📖 Помощь по использованию бота:

Просто отправьте мне любой вопрос, и я найду ответ в базе знаний.

⚡ Команды:
/start - Приветственное сообщение
/help - Эта справка
/info - Информация о системе
/stats - Статистика работы бота

⚠️ Ограничения:
• Максимальная длина ответа: 4000 символов
• Лимит запросов: 5 в минуту
• Ответы основаны только на информации из базы знаний

Если я не могу найти ответ в базе знаний, я честно скажу "не знаю".
        """
        await update.message.reply_text(help_text)

    async def info_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик команды /info

        Args:
            update: Объект обновления Telegram
            context: Контекст бота
        """
        info = self.pipeline.get_pipeline_info()

        info_text = f"""
🔧 Информация о системе:

🤖 Модель LLM: {info['llm_model']}
🧠 Модель эмбеддингов: {info['embedding_model']}
📊 Few-shot примеров: {info['few_shot_examples']}
        """

        if "error" not in info['vector_store']:
            info_text += f"\n📚 Векторная база: {info['vector_store']['name']}"
            info_text += f"\n📄 Документов в базе: {info['vector_store']['count']}"
        else:
            info_text += "\n📚 Векторная база: Не доступна"

        await update.message.reply_text(info_text)

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик команды /stats

        Args:
            update: Объект обновления Telegram
            context: Контекст бота
        """
        stats_text = f"""
📊 Статистика бота:

Всего запросов: {self.stats["total_requests"]}
Успешных ответов: {self.stats["successful_responses"]}
Ошибок: {self.stats["errors"]}

Активных пользователей (за последний час): {len(self.user_requests)}
        """
        await update.message.reply_text(stats_text)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик текстовых сообщений

        Args:
            update: Объект обновления Telegram
            context: Контекст бота
        """
        user_id = update.effective_user.id
        message_text = update.message.text

        # Обновляем статистику
        self.stats["total_requests"] += 1

        # Проверяем лимит запросов
        if not self._check_rate_limit(user_id):
            await update.message.reply_text(
                "⚠️ Слишком много запросов. Пожалуйста, подождите минуту."
            )
            return

        # Отправляем сообщение о обработке
        processing_msg = await update.message.reply_text("🔍 Ищу информацию...")

        try:
            # Обрабатываем запрос через RAG пайплайн
            response = self.pipeline.process_query(message_text)

            # Формируем ответ
            if response.has_context:
                answer = f"💡 {response.answer}"
                if response.used_sources:
                    answer += f"\n\n📚 Источники: {', '.join(response.used_sources)}"
            else:
                answer = f"❌ {response.answer}"

            # Добавляем время обработки
            answer += f"\n\n⏱ Время обработки: {response.processing_time:.2f} сек"

            # Обрезаем ответ, если он слишком длинный
            if len(answer) > self.bot_config.max_message_length:
                answer = answer[:self.bot_config.max_message_length - 100] + "...\n\n[Ответ был обрезан из-за ограничений Telegram]"

            # Обновляем сообщение с ответом
            await processing_msg.edit_text(answer)
            self.stats["successful_responses"] += 1

        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}")
            await processing_msg.edit_text(
                f"⚠️ Произошла ошибка при обработке запроса: {str(e)}"
            )
            self.stats["errors"] += 1

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработчик ошибок

        Args:
            update: Объект обновления Telegram
            context: Контекст бота
        """
        try:
            # Логируем ошибку
            logger.error("Exception while handling an update:", exc_info=context.error)

            # Отправляем сообщение об ошибке пользователю, если возможно
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    "⚠️ Произошла внутренняя ошибка. Пожалуйста, попробуйте позже."
                )
        except Exception as e:
            logger.error(f"Ошибка в обработчике ошибок: {e}")

    def setup_handlers(self):
        """Настройка обработчиков команд"""
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("info", self.info_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))

        # Обработчик текстовых сообщений
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

        # Обработчик ошибок
        self.application.add_error_handler(self.error_handler)

    async def run(self):
        """
        Запуск бота

        Запуск в асинхронном режиме
        """
        # Создаем приложение
        self.application = Application.builder().token(self.bot_config.token).build()

        # Настраиваем обработчики
        self.setup_handlers()

        # Запускаем бота
        logger.info("Бот запущен...")
        await self.application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


def run_telegram_bot(token: str):
    """
    Запуск Telegram бота

    Args:
        token: Токен Telegram бота
    """
    # Конфигурация бота
    bot_config = BotConfig(
        token=token,
        admin_ids=[],  # Можно добавить ID администраторов
        rate_limit_per_user=5,
        max_message_length=4000
    )

    # Создаем и запускаем бота
    bot = RAGTelegramBot(bot_config)

    # Запускаем бота
    asyncio.run(bot.run())


def main():
    """Точка входа для Telegram бота"""
    import argparse

    parser = argparse.ArgumentParser(description='Запуск RAG Telegram бота')
    parser.add_argument('--token', required=True, help='Токен Telegram бота')

    args = parser.parse_args()

    # Запускаем бота
    run_telegram_bot(args.token)


if __name__ == "__main__":
    main()