import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DATA_PATH = Path("../processed/legal_corpus.csv")
DEFAULT_VECTOR_DIR = Path("../vector_store")
REQUIRED_COLUMNS = ["id", "content"]
DEFAULT_METADATA_COLUMNS = [
    "id",
    "doc_type",
    "year",
    "article_no",
    "article_title",
]


def validate_input(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"CSV thiếu cột bắt buộc: {missing}")


def resolve_metadata_columns(df: pd.DataFrame, requested_columns: list[str] | None) -> list[str]:
    if requested_columns:
        columns = [column.strip() for column in requested_columns if column.strip()]
    else:
        columns = DEFAULT_METADATA_COLUMNS
    return [column for column in columns if column in df.columns]


def build_index(embeddings: np.ndarray, use_cosine: bool) -> faiss.Index:
    dimension = embeddings.shape[1]
    if use_cosine:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dimension)
    else:
        index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.Index, index_path: Path) -> None:
    try:
        faiss.write_index(index, str(index_path))
        return
    except RuntimeError:
        serialized = faiss.serialize_index(index)
        with index_path.open("wb") as f:
            f.write(serialized.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS vector DB from legal_corpus.csv")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Input CSV path")
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR, help="Output vector store directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding")
    parser.add_argument("--no-cosine", action="store_true", help="Use L2 distance instead of cosine similarity")
    parser.add_argument(
        "--metadata-columns",
        nargs="*",
        default=None,
        help="Optional metadata columns to save (default is compact legal columns)",
    )
    args = parser.parse_args()

    data_path = args.data_path.resolve()
    vector_dir = args.vector_dir.resolve()
    vector_dir.mkdir(parents=True, exist_ok=True)

    index_path = vector_dir / "index.faiss"
    meta_path = vector_dir / "metadata.json"

    print("Loading dataset...")
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {data_path}")

    df = pd.read_csv(data_path)
    validate_input(df)

    texts = df["content"].fillna("").astype(str).str.strip()
    texts = texts[texts != ""]
    if texts.empty:
        raise ValueError("Không có nội dung hợp lệ trong cột content để tạo embedding")

    df = df.loc[texts.index].reset_index(drop=True)
    texts_list = texts.tolist()
    print(f"Loaded {len(texts_list)} chunks")

    print("Loading embedding model...")
    model = SentenceTransformer(args.model)
    print("Model loaded")

    print("Creating embeddings...")
    embeddings = model.encode(
        texts_list,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    print("Building FAISS index...")
    use_cosine = not args.no_cosine
    index = build_index(embeddings, use_cosine=use_cosine)
    print(f"Total vectors: {index.ntotal}")

    save_faiss_index(index, index_path)
    print(f"FAISS index saved: {index_path}")

    metadata_columns = resolve_metadata_columns(df, args.metadata_columns)
    if "id" not in metadata_columns and "id" in df.columns:
        metadata_columns.insert(0, "id")
    metadata = df[metadata_columns].to_dict(orient="records") if metadata_columns else []

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": args.model,
                "metric": "cosine" if use_cosine else "l2",
                "vector_count": int(index.ntotal),
                "metadata_columns": metadata_columns,
                "records": metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Metadata saved: {meta_path}")
    print("\nVector database build complete!")


if __name__ == "__main__":
    main()