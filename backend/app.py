from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.rag_engine import LegalRAGEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(title="VNLaw Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str


engine = LegalRAGEngine(project_root=PROJECT_ROOT)


@app.get("/")
def root() -> dict:
    return {
        "service": "vnlaw-backend-api",
        "health": "/api/health",
        "chat": "/api/chat",
        "docs": "/docs",
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "embedding_model": engine.embedding_model_name,
        "vector_count": int(engine.index.ntotal),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        contexts = engine.search_law(question=question, top_k=payload.top_k)
        answer = engine.generate_answer(question=question, contexts=contexts)
        return ChatResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
