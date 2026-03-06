import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# =============================
# CONFIG
# =============================

INDEX_PATH = Path("../vector_store/index.faiss")
METADATA_PATH = Path("../vector_store/metadata.json")
DATA_PATH = Path("../processed/legal_corpus.csv")

TOP_K = 5

# OpenRouter client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
) if os.getenv("OPENROUTER_API_KEY") else None


def load_faiss_index(index_path: Path) -> faiss.Index:
    resolved = index_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Không tìm thấy FAISS index: {resolved}")

    try:
        return faiss.read_index(str(resolved))
    except RuntimeError:
        raw = resolved.read_bytes()
        arr = np.frombuffer(raw, dtype=np.uint8)
        return faiss.deserialize_index(arr)

# =============================
# LOAD VECTOR DB
# =============================

print("Loading FAISS index...")
index = load_faiss_index(INDEX_PATH)

print("Loading metadata...")
with METADATA_PATH.open("r", encoding="utf-8") as f:
    payload = json.load(f)

if isinstance(payload, dict):
    metadata = payload.get("records", [])
    embedding_model_name = payload.get(
        "embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    metric = payload.get("metric", "l2")
else:
    metadata = payload
    embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    metric = "l2"

if not isinstance(metadata, list):
    raise ValueError("metadata.json không đúng định dạng")

if metadata and "content" not in metadata[0]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy CSV: {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    if "id" not in df.columns or "content" not in df.columns:
        raise ValueError("legal_corpus.csv phải có cột id và content")

    id_to_content = dict(
        zip(df["id"].astype(str), df["content"].fillna("").astype(str))
    )
    for row in metadata:
        row_id = str(row.get("id", ""))
        row["content"] = id_to_content.get(row_id, "")

print("Loading embedding model...")
model = SentenceTransformer(embedding_model_name)

print("System ready!\n")

# =============================
# SEARCH
# =============================

def search_law(question):
    embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=(metric == "cosine"),
    ).astype("float32")

    if embedding.shape[1] != index.d:
        raise ValueError(
            f"Lệch chiều vector: query_dim={embedding.shape[1]} nhưng index_dim={index.d}. "
            f"Hãy build lại index hoặc dùng đúng embedding model."
        )

    _, I = index.search(np.array(embedding).astype("float32"), TOP_K)

    contexts = []

    for idx in I[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        row = metadata[idx]
        content = row.get("content") or row.get("text") or ""
        if not content:
            continue
        cite = f"[{row.get('doc_type', '')} {row.get('year', '')} Điều {row.get('article_no', '')}]"
        contexts.append(f"{cite}\n{content}")

    return contexts


# =============================
# GENERATE ANSWER
# =============================

def generate_answer(question, contexts):
    if client is None:
        return "Thiếu OPENROUTER_API_KEY trong .env."

    context_text = "\n\n".join(contexts)

    prompt = f"""
Bạn là chatbot hỗ trợ tư vấn pháp luật Việt Nam.

Dựa trên các đoạn luật sau:

{context_text}

Hãy trả lời câu hỏi của người dùng rõ ràng và chính xác.

Câu hỏi:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Bạn là trợ lý pháp luật Việt Nam."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# =============================
# MAIN
# =============================

while True:

    question = input("\nNhập câu hỏi (exit để thoát): ")

    if question.lower() == "exit":
        break

    print("\n🔎 Đang tìm luật liên quan...")

    contexts = search_law(question)

    if not contexts:
        print("\nKhông tìm thấy điều luật phù hợp trong vector DB.")
        continue

    print("\n📄 Top luật tìm được:\n")

    for i, c in enumerate(contexts):
        print(f"{i+1}. {c[:200]}...\n")

    print("🤖 AI đang trả lời...\n")

    answer = generate_answer(question, contexts)

    print("📌 Trả lời:\n")
    print(answer)