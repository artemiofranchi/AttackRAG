from __future__ import annotations

from dataclasses import dataclass

from attackrag.documents import RawDocument


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def chunk_documents(
    documents: list[RawDocument],
    *,
    max_chars: int = 1200,
    overlap: int = 80,
) -> list[Chunk]:
    if overlap >= max_chars:
        raise ValueError("overlap должен быть меньше max_chars")
    chunks: list[Chunk] = []
    for doc in documents:
        text = doc.text
        start = 0
        part = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            piece = text[start:end].strip()
            if piece:
                cid = f"{doc.doc_id}__{part}"
                chunks.append(Chunk(chunk_id=cid, doc_id=doc.doc_id, text=piece))
                part += 1
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks
