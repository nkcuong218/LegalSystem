from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import APIError, AuthenticationError, OpenAI, RateLimitError
from sentence_transformers import SentenceTransformer

load_dotenv()


@dataclass
class RetrievedContext:
    citation: str
    content: str


class LegalRAGEngine:
    def __init__(self, project_root: Path, top_k: int = 5) -> None:
        self.project_root = project_root
        self.top_k = top_k

        self.index_path = self.project_root / "vector_store" / "index.faiss"
        self.metadata_path = self.project_root / "vector_store" / "metadata.json"
        self.data_path = self.project_root / "processed" / "legal_corpus.csv"

        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
        self.client = (
            OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            if api_key
            else None
        )

        self.index = self._load_faiss_index(self.index_path)
        self.metadata, self.embedding_model_name, self.metric = self._load_metadata(self.metadata_path)
        self._hydrate_content_if_needed()
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def _load_faiss_index(self, index_path: Path) -> faiss.Index:
        if not index_path.exists():
            raise FileNotFoundError(f"Không tìm thấy FAISS index: {index_path}")

        try:
            return faiss.read_index(str(index_path))
        except RuntimeError:
            raw = index_path.read_bytes()
            arr = np.frombuffer(raw, dtype=np.uint8)
            return faiss.deserialize_index(arr)

    def _load_metadata(self, metadata_path: Path) -> tuple[list[dict[str, Any]], str, str]:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Không tìm thấy metadata: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            metadata = payload.get("records", [])
            embedding_model_name = payload.get(
                "embedding_model",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
            metric = payload.get("metric", "l2")
        else:
            metadata = payload
            embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            metric = "l2"

        if not isinstance(metadata, list):
            raise ValueError("metadata.json không đúng định dạng")

        return metadata, embedding_model_name, metric

    def _hydrate_content_if_needed(self) -> None:
        if not self.metadata:
            return

        if "content" in self.metadata[0]:
            return

        if not self.data_path.exists():
            raise FileNotFoundError(f"Không tìm thấy CSV: {self.data_path}")

        df = pd.read_csv(self.data_path)
        if "id" not in df.columns or "content" not in df.columns:
            raise ValueError("legal_corpus.csv phải có cột id và content")

        id_to_content = dict(zip(df["id"].astype(str), df["content"].fillna("").astype(str)))
        for row in self.metadata:
            row["content"] = id_to_content.get(str(row.get("id", "")), "")

    def search_law(self, question: str, top_k: int | None = None) -> list[RetrievedContext]:
        k = top_k or self.top_k

        embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=(self.metric == "cosine"),
        ).astype("float32")

        if embedding.shape[1] != self.index.d:
            raise ValueError(
                f"Lệch chiều vector: query_dim={embedding.shape[1]} nhưng index_dim={self.index.d}."
            )

        _, indices = self.index.search(np.array(embedding).astype("float32"), k)

        contexts: list[RetrievedContext] = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.metadata):
                continue

            row = self.metadata[idx]
            content = row.get("content") or row.get("text") or ""
            if not content:
                continue

            citation = f"[{row.get('doc_type', '')} {row.get('year', '')} Điều {row.get('article_no', '')}]"
            contexts.append(RetrievedContext(citation=citation, content=content))

        return contexts

    def generate_answer(self, question: str, contexts: list[RetrievedContext]) -> str:
        if not contexts:
            return "Không tìm thấy điều luật phù hợp trong vector DB."

        if self.client is None:
            context_text = "\n\n".join([f"{ctx.citation}\n{ctx.content}" for ctx in contexts])
            return (
                "Thiếu OPENROUTER_API_KEY trong .env. Đang trả về các đoạn luật truy xuất được:\n\n"
                f"{context_text}"
            )

        context_text = "\n\n".join([f"{ctx.citation}\n{ctx.content}" for ctx in contexts])

        prompt = f"""
Bạn là chatbot hỗ trợ tư vấn pháp luật Việt Nam.

Dựa trên các đoạn luật sau:

{context_text}

Hãy trả lời câu hỏi của người dùng rõ ràng và chính xác.

Câu hỏi:
{question}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.openrouter_model,
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý pháp luật Việt Nam."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or "Không có phản hồi từ mô hình."
        except AuthenticationError:
            return "Lỗi xác thực OpenRouter. Kiểm tra lại OPEN_ROUTER_API_KEY trong .env."
        except RateLimitError:
            return "OpenRouter đang giới hạn tốc độ yêu cầu. Vui lòng thử lại sau."
        except APIError as exc:
            return f"Lỗi API từ OpenRouter: {exc}"
        except Exception as exc:
            return f"Không thể gọi LLM: {exc}"
