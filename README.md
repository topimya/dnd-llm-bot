# D&D Rules Bot

A Telegram bot that answers Dungeons & Dragons 5e rules questions using a PDF you provide. It retrieves relevant passages with FAISS and generates an English answer with a local Mistral model.

## How it works

1. At startup, pdfplumber extracts text from the selected PDF pages. For two-column pages, it reads the left half before the right half.
2. The bot splits the text into overlapping passages and embeds them with multilingual MiniLM. A FAISS flat index keeps those vectors in memory for exact L2 nearest-neighbor search.
3. For each question, the bot embeds the question and retrieves up to five passages. It adds passages that fit the prompt budget, reserving space for the model's reply.
4. llama-cpp-python runs the local GGUF model. The prompt asks for an English answer based on the excerpts and asks the model to say when the excerpts lack enough information. The bot sends the result back to Telegram.

## Run locally

You'll need Python 3.11, a Telegram bot token from BotFather, a text-based D&D 5e rules PDF you can use, and an instruction-tuned GGUF file with an embedded chat template. The original model was Mistral 7B Instruct v0.2 in Q4_K_M format. The PDF and model weights are supplied separately.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:TELEGRAM_BOT_TOKEN = Read-Host 'Telegram bot token' -MaskInput
.\.venv\Scripts\python.exe bot.py --pdf "path/to/rules.pdf" --model "path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
```

Use PowerShell 7 for the masked token prompt. Keep the token out of source control. Installation of llama-cpp-python may require a compatible wheel or a C++ build toolchain. The first run may download the embedding model. Telegram polling requires a network connection. PDF indexing and model inference run on your machine.

The default starts at the first PDF page and assumes two columns. Use `--columns 1` for single-column text or `--start-page 4` to start on the fourth PDF page. Run `python bot.py --help` to see all options.

In Telegram, send `/start` and try *How does a character make a saving throw?* Use a PDF that covers the question.

## Prototype limits

- The index is rebuilt at startup and reused until the process exits. Telegram updates are handled one at a time so the shared model processes one request at a time.
- The PDF parser expects selectable text and a consistent layout. Scans need OCR, and tables or mixed layouts may extract out of order.
- Answers depend on retrieval and generation quality. The bot does not show page citations or retain conversation history.
- Local inference time depends on the machine and model. Answer quality and latency have not been benchmarked.
