import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import time 

MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def extract_page_by_halves(page):
    w, h = page.width, page.height
    left = page.crop((0, 0, w/2, h)).extract_text() or ""
    right = page.crop((w/2, 0, w, h)).extract_text() or ""
    # читать сначала левая колонка, потом правая
    return (left + " " + right).strip()

def str_text(file_pdf):
    texts = []
    with pdfplumber.open(file_pdf) as pdf:
        for i in range(3, len(pdf.pages)):
            texts.append(extract_page_by_halves(pdf.pages[i]))

    full_text = " ".join(texts).replace('\n',' ')
    return full_text

def chunk_text(text, max_words=200, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_words]
        chunks.append(' '.join(chunk))
        i += max_words - overlap
    return chunks

def embedding_chunk(chunks):
    embeddings = MODEL.encode(chunks, convert_to_numpy=True)

    d = embeddings.shape[1] 
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings, dtype=np.float32))

    return index

def save_data(index, chunks, index_path="dnd_index.faiss", chunks_path="dnd_chunks.pkl"):
    faiss.write_index(index, index_path)

    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    return 

def load_data(iindex_path="dnd_index.faiss", chunks_path="dnd_chunks.pkl"):
    index = faiss.read_index(iindex_path)

    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)

    return index, chunks

def search_question(question, index, chunks, k=5):
    question_embedding = MODEL.encode([question], convert_to_numpy=True)
    D, I = index.search(np.array(question_embedding, dtype=np.float32), k=k)

    relevant_chuks = [chunks[i] for i in I[0]]

    return relevant_chuks

def generate_answer_llm(question, context, llm):
    prompt = f"""Ты выступаешь как эксперт по Dungeons & Dragons 5e.
    Тебе дан текст правил. Отвечай строго, используя его, не придумывай от себя. Пиши СТРОГО на русском языке, переводя термины и названия.
    Если видишь английский текст — переводи его. 
    Никогда не используй английский в ответе.

    Контекст:
    {context}

    Вопрос:
    {question}

    Формат ответа:
    1. Краткое описание
    2. Подробности
    3. Примеры (если есть)"""

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Ты умный ассистент, отвечающий строго по правилам D&D 5e. Отвечай строго на русском языке. Не добавляй ничего своего. Не придумывай имена, цифры или новые детали."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=600,
        temperature=0.5,
    )
    return response["choices"][0]["message"]["content"]

def get_answer(question, llm):
    index_path = "dnd_index.faiss"
    chunks_path = "dnd_chunks.pkl"

    if os.path.exists(index_path) and os.path.exists(chunks_path):
        print("Файлы найдены. Загружаем индекс и чанки...")
        index, chunks = load_data(index_path, chunks_path)
    else:
        print("Файлы не найдены. Генерируем заново...")
        
        text = str_text('C:/Users/konki/Desktop/all/llm/qa_pdf/5e Players Handbook.pdf')
        chunks = chunk_text(text)
        index = embedding_chunk(chunks)
        save_data(index, chunks, index_path, chunks_path)
    start_time = time.time() 
    relevant_chunks = search_question(question, index, chunks, k=5)
    end_time = time.time()
    print("=== Время обработки вопроса до llm: {:.2f} секунд ===".format(end_time - start_time))

    start_time = time.time() 
    answer = generate_answer_llm(question, relevant_chunks, llm)
    end_time = time.time() 
    print("=== Время обработки запроса: {:.2f} секунд ===".format(end_time - start_time))
    return answer

    
