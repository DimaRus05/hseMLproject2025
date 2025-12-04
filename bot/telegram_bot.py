import argparse
import os
import joblib
import re
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _shorten(text: str, max_len: int = 400) -> str:
    if not text:
        return ''
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 3] + '...'


def _format_reply_with_text(text: str, prob: float, prefix: str = 'Это сообщение') -> str:
    short = _shorten(text)
    pct = round(100 * float(prob), 1)
    if prob >= 0.5:
        if short:
            return f"Сообщение '{short}' токсично на {pct} процентов"
        return f"Сообщение токсично на {pct} процентов"
    else:
        if short:
            return f"Сообщение '{short}' не токсично"
        return "Сообщение не токсично"


def load_model(path):
    if not os.path.exists(path):
        raise RuntimeError(f'Model not found: {path}')
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f'Failed to load model: {e}')


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Пришлите мне сообщение, и я скажу насколько оно токсично')


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('/start - start\n/ping - health check\nПришлите мне сообщение в ответ на которое хотите получить оценку токсичности.')


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('pong')



async def check_reply_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if msg is None:
        logger.debug('check_reply_handler: no message')
        return
    trigger_text = (msg.text or msg.caption or '')
    if msg.chat.type == 'private':
        logger.debug('check_reply_handler: private message received but will be handled by private handler')
        return

    if not trigger_text or 'токс' not in trigger_text.lower():
        logger.debug('group message without trigger; ignoring')
        return

    if msg.reply_to_message is None:
        logger.debug('trigger present but no reply target; ignoring')
        return

    target = msg.reply_to_message
    text = target.text or target.caption or ''
    if not text or not text.strip():
        try:
            await msg.reply_text('Нельзя проверить: сообщение не содержит текста.')
        except Exception as e:
            logger.exception('check_reply_handler: failed to send empty-target reply: %s', e)
        return

    model = context.bot_data.get('model')
    if model is None:
        try:
            await msg.reply_text('Модель не загружена.')
        except Exception as e:
            logger.exception('check_reply_handler: failed to send model-missing reply: %s', e)
        logger.warning('check_reply_handler: model not loaded')
        return

    try:
        prob = model.predict_proba([text])[0][1]
    except Exception as e:
        logger.exception('predict_proba failed in check handler: %s', e)
        try:
            pred = model.predict([text])[0]
            prob = 1.0 if pred == 1 else 0.0
        except Exception as e2:
            logger.exception('predict fallback failed in check handler: %s', e2)
            try:
                await msg.reply_text('Ошибка при инференсе модели.')
            except Exception:
                pass
            return

    reply_text = _format_reply_with_text(text, prob)
    try:
        await msg.reply_text(reply_text)
    except Exception as e:
        logger.exception('check_reply_handler: failed to send result reply: %s', e)


async def private_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle private (one-to-one) messages: respond to every text message with toxicity score."""
    msg = update.message
    if msg is None:
        return
    text_in = (msg.text or msg.caption or '')
    if not text_in or text_in.startswith('/'):
        return

    model = context.bot_data.get('model')
    if model is None:
        try:
            await msg.reply_text('Модель не загружена.')
        except Exception as e:
            logger.exception('private_message_handler: failed to send model-missing reply: %s', e)
        return

    try:
        prob = model.predict_proba([text_in])[0][1]
    except Exception as e:
        logger.exception('predict_proba failed in private handler: %s', e)
        try:
            pred = model.predict([text_in])[0]
            prob = 1.0 if pred == 1 else 0.0
        except Exception as e2:
            logger.exception('predict fallback failed in private handler: %s', e2)
            try:
                await msg.reply_text('Ошибка при инференсе модели.')
            except Exception:
                pass
            return

    reply_text = _format_reply_with_text(text_in, prob)
    try:
        await msg.reply_text(reply_text)
    except Exception as e:
        logger.exception('private_message_handler: failed to send result reply: %s', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', default=None, help='Telegram bot token (or set TELEGRAM_TOKEN env var)')
    parser.add_argument('--model', default='models/calibrated_model.joblib', help='Path to calibrated model')
    args = parser.parse_args()

    load_dotenv()

    token = args.token or os.environ.get('TELEGRAM_TOKEN')
    if not token:
        raise RuntimeError('Telegram token missing: set TELEGRAM_TOKEN or pass --token')
    model = load_model(args.model)

    app = ApplicationBuilder().token(token).build()
    app.bot_data['model'] = model
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_cmd))
    app.add_handler(CommandHandler('ping', ping))
    # Handle private messages: respond to every text message in direct chats
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & filters.TEXT, private_message_handler))
    app.add_handler(MessageHandler(filters.REPLY & filters.TEXT, check_reply_handler))

    logger.info('Bot starting (polling)...')
    app.run_polling()


if __name__ == '__main__':
    main()
