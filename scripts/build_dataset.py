from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


HEADER_PATTERNS = {
    "part": re.compile(r"^(Phần\s+thứ\s+.+|PHẦN\s+.+)$", re.IGNORECASE),
    "chapter": re.compile(r"^Chương\s+([IVXLC0-9]+)\s*$", re.IGNORECASE),
    "section": re.compile(r"^Mục\s+([IVXLC0-9]+|\d+)\.?\s*(.*)$", re.IGNORECASE),
    "article": re.compile(r"^Điều\s+(\d+[A-Za-z]?)\.\s*(.*)$", re.IGNORECASE),
}


SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.!?;:])\s+")

OUTPUT_COLUMNS = [
    "id",
    "content",
    "doc_type",
    "year",
    "article_no",
    "article_title",
]


@dataclass
class ChunkRecord:
    id: str
    source_path: str
    category: str
    doc_type: str
    doc_no: str
    year: str
    short_name: str
    part: str
    chapter: str
    section: str
    article_no: str
    article_title: str
    chunk_index: int
    content: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_path": self.source_path,
            "category": self.category,
            "doc_type": self.doc_type,
            "doc_no": self.doc_no,
            "year": self.year,
            "short_name": self.short_name,
            "part": self.part,
            "chapter": self.chapter,
            "section": self.section,
            "article_no": self.article_no,
            "article_title": self.article_title,
            "chunk_index": self.chunk_index,
            "content": self.content,
        }


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def normalize_block(text: str) -> str:
    lines = [normalize_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def parse_filename(path: Path) -> dict:
    stem = path.stem
    parts = stem.split("_")
    doc_type = parts[0] if len(parts) >= 1 else "UNKNOWN"
    doc_no = parts[1] if len(parts) >= 2 else ""
    year = parts[2] if len(parts) >= 3 else ""
    short_name = "_".join(parts[3:]) if len(parts) >= 4 else stem
    return {
        "doc_type": doc_type,
        "doc_no": doc_no,
        "year": year,
        "short_name": short_name,
    }


def smart_split(text: str, max_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    units = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        sentences = SENTENCE_SPLIT_RE.split(para)
        units.extend([s.strip() for s in sentences if s.strip()])

    if not units:
        units = [text]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for unit in units:
        extra = len(unit) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)

            if overlap > 0 and chunk:
                tail = chunk[-overlap:]
                current = [tail, unit]
                current_len = len(tail) + 1 + len(unit)
            else:
                current = [unit]
                current_len = len(unit)
        else:
            current.append(unit)
            current_len += extra

    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def parse_document(path: Path, max_chars: int, overlap: int) -> List[ChunkRecord]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.rstrip() for line in text.splitlines()]
    meta = parse_filename(path)

    category = path.parent.name
    part = ""
    chapter = ""
    section = ""
    article_no = ""
    article_title = ""
    article_buffer: List[str] = []
    doc_prefix_lines: List[str] = []
    records: List[ChunkRecord] = []

    def flush_article() -> None:
        nonlocal article_buffer, article_no, article_title
        if not article_buffer:
            return

        article_text = normalize_block("\n".join(article_buffer))
        if not article_text:
            article_buffer = []
            return

        heading = []
        if part:
            heading.append(part)
        if chapter:
            heading.append(chapter)
        if section:
            heading.append(section)
        if article_no:
            heading.append(f"Điều {article_no}")
        if article_title:
            heading.append(article_title)

        full_text = "\n".join([" - ".join(heading), article_text]) if heading else article_text
        chunks = smart_split(full_text, max_chars=max_chars, overlap=overlap)

        for idx, chunk in enumerate(chunks, start=1):
            rec_id = (
                f"{path.stem}|dieu_{article_no or 'NA'}|c{idx}"
                if article_no
                else f"{path.stem}|chunk_{idx}"
            )
            records.append(
                ChunkRecord(
                    id=rec_id,
                    source_path=str(path.as_posix()),
                    category=category,
                    doc_type=meta["doc_type"],
                    doc_no=meta["doc_no"],
                    year=meta["year"],
                    short_name=meta["short_name"],
                    part=part,
                    chapter=chapter,
                    section=section,
                    article_no=article_no,
                    article_title=article_title,
                    chunk_index=idx,
                    content=chunk,
                )
            )

        article_buffer = []

    for raw in lines:
        line = normalize_line(raw)
        if not line:
            continue

        part_match = HEADER_PATTERNS["part"].match(line)
        chapter_match = HEADER_PATTERNS["chapter"].match(line)
        section_match = HEADER_PATTERNS["section"].match(line)
        article_match = HEADER_PATTERNS["article"].match(line)

        if article_match:
            flush_article()
            article_no = article_match.group(1).strip()
            article_title = article_match.group(2).strip()
            article_buffer = []
            continue

        if part_match:
            flush_article()
            part = line
            continue

        if chapter_match:
            flush_article()
            chapter = f"Chương {chapter_match.group(1).strip()}"
            section = ""
            continue

        if section_match:
            flush_article()
            suffix = section_match.group(2).strip()
            section = f"Mục {section_match.group(1).strip()}"
            if suffix:
                section = f"{section}. {suffix}"
            continue

        if article_no:
            article_buffer.append(line)
        else:
            doc_prefix_lines.append(line)

    flush_article()

    if not records and doc_prefix_lines:
        fallback_text = normalize_block("\n".join(doc_prefix_lines))
        chunks = smart_split(fallback_text, max_chars=max_chars, overlap=overlap)
        for idx, chunk in enumerate(chunks, start=1):
            records.append(
                ChunkRecord(
                    id=f"{path.stem}|preface_{idx}",
                    source_path=str(path.as_posix()),
                    category=category,
                    doc_type=meta["doc_type"],
                    doc_no=meta["doc_no"],
                    year=meta["year"],
                    short_name=meta["short_name"],
                    part=part,
                    chapter=chapter,
                    section=section,
                    article_no="",
                    article_title="",
                    chunk_index=idx,
                    content=chunk,
                )
            )

    return records


def write_csv(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return 0

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def select_output_fields(row: dict) -> dict:
    return {key: row.get(key, "") for key in OUTPUT_COLUMNS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build legal corpus dataset for RAG chatbot.")
    parser.add_argument("--input-dir", default="../raw_data", help="Path to raw txt documents")
    parser.add_argument("--output-dir", default="../processed", help="Output folder")
    parser.add_argument("--max-chars", type=int, default=1200, help="Max characters per chunk")
    parser.add_argument("--overlap", type=int, default=180, help="Character overlap between chunks")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No txt files found under: {input_dir}")

    all_records: List[ChunkRecord] = []
    for txt_path in txt_files:
        records = parse_document(txt_path, max_chars=args.max_chars, overlap=args.overlap)
        all_records.extend(records)

    dataset_rows = [select_output_fields(record.to_dict()) for record in all_records]

    csv_path = output_dir / "legal_corpus.csv"
    csv_count = write_csv(csv_path, dataset_rows)

    print(f"Processed {len(txt_files)} files")
    print(f"Generated {csv_count} chunks")
    print(f"CSV:     {csv_path}")


if __name__ == "__main__":
    main()
