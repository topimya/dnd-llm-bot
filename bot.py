from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from llama_cpp import Llama
from app.pdf_read import get_answer
import time

start_time = time.time() 
llm = Llama("weights/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=4096, verbose=False)
end_time = time.time()
print("=== Время обработки модели: {:.2f} секунд ===".format(end_time - start_time))

TOKEN = 'YOUR TOKEN'

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я D&D помошник. Задавай свой вопрос. 🎲")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    await update.message.reply_text("Обработка запроса... ⏳. Время ожидания ~3 мин.")
    answer = get_answer(question, llm)
    add_answer = '🎲🏰 Ответ на вопрос 🏰🎲\n'
    answer = add_answer + '\n' + answer
    await update.message.reply_text(answer)

if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT  &  ~filters.COMMAND, handle_message))

    print('Бот запущен...')
    app.run_polling()


