import argparse
import asyncio
import logging
import os
from pathlib import Path

from llama_cpp import Llama
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from app.pdf_read import RulesCorpus

LOGGER = logging.getLogger(__name__)
# Every Unicode code point takes at most two UTF-16 units.
REPLY_CHUNK_SIZE = 2048


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_message:
        await update.effective_message.reply_text(
            "Hi! Ask me a question about Dungeons & Dragons 5e rules."
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if not message or not message.text:
        return

    question = message.text.strip()
    if not question:
        await message.reply_text("Please enter a rules question.")
        return

    await message.reply_text("Looking up the rules and generating an answer...")
    try:
        corpus = context.bot_data["corpus"]
        answer = await asyncio.to_thread(corpus.answer, question, context.bot_data["llm"])
    except Exception:
        LOGGER.exception("Could not answer the rules question")
        await message.reply_text("I could not answer right now. Please try again.")
        return

    for offset in range(0, len(answer), REPLY_CHUNK_SIZE):
        await message.reply_text(answer[offset:offset + REPLY_CHUNK_SIZE])


def main():
    parser = argparse.ArgumentParser(description="Run the D&D Rules Bot.")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to a text-based rules PDF")
    parser.add_argument("--model", required=True, type=Path, help="Path to an instruct GGUF model")
    parser.add_argument("--columns", type=int, choices=(1, 2), default=2)
    parser.add_argument("--start-page", type=int, default=1, help="First PDF page to index (1-based)")
    args = parser.parse_args()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        parser.error("Set the TELEGRAM_BOT_TOKEN environment variable.")
    for path in (args.pdf, args.model):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if args.start_page < 1:
        parser.error("--start-page must be at least 1.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    app = ApplicationBuilder().token(token).concurrent_updates(False).build()
    LOGGER.info("Indexing the rules PDF")
    app.bot_data["corpus"] = RulesCorpus(args.pdf, args.columns, args.start_page)
    LOGGER.info("Loading the local language model")
    app.bot_data["llm"] = Llama(model_path=str(args.model), n_ctx=4096, verbose=False)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    LOGGER.info("Bot is ready")
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
