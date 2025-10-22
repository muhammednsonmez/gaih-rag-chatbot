#!/usr/bin/env python3
"""Simple PDF -> Chroma ingestion script with multilingual E5 embeddings."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch

try:
    from pypdf import PdfReader
except Exception as exc:  # pragma: no cover - configuration issue
    raise RuntimeError("pypdf bulunamadı. 'pip install pypdf' ile kur.") from exc

try:
    import fitz  # type: ignore

    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

# --------------------------
# Config
# --------------------------
DATA_DIR = Path("data")
VECTOR_DIR = Path("vectordb")
COLLECTION_NAME = "docs"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

DEFAULT_CHUNK_SIZE = 900
DEFAULT_OVERLAP = 200
DEFAULT_MAX_WORKERS = min(6, (os.cpu_count() or 4))
DEFAULT_ADD_BATCH = 512
DEFAULT_EMBED_BATCH_CPU = 32
DEFAULT_EMBED_BATCH_GPU = 128

_SOFT_HYPHEN = "\u00ad"
_DEHYPHEN_RE = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_SQUEEZE_WS_RE = re.compile(r"[ \t]+")


# --------------------------
# Helpers
# --------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace(_SOFT_HYPHEN, "")
    text = text.replace("\u2010", "-")
    text = _DEHYPHEN_RE.sub("", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    text = _SQUEEZE_WS_RE.sub(" ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def _extract_pdf_pymupdf(path: Path) -> str:
    pieces: List[str] = []
    with fitz.open(str(path)) as doc:  # type: ignore
        for page in doc:
            try:
                pieces.append(clean_text(page.get_text("text") or ""))
            except Exception:
                pieces.append("")
    return clean_text("\n\n".join(pieces))


def _extract_pdf_pypdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pieces: List[str] = []
    for page in reader.pages:
        try:
            pieces.append(clean_text(page.extract_text() or ""))
        except Exception:
            pieces.append("")
    return clean_text("\n\n".join(pieces))


def extract_pdf(path: Path) -> str:
    if HAS_PYMUPDF:
        return _extract_pdf_pymupdf(path)
    return _extract_pdf_pypdf(path)


def hash_chunk(text: str, source: str, index: int) -> str:
    h = hashlib.sha1()
    h.update(source.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(index).encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def load_pdf_chunks(path: Path, chunk_size: int, overlap: int) -> List[Tuple[str, dict, str]]:
    try:
        full_text = extract_pdf(path)
    except Exception as exc:
        print(f"[WARN] PDF okunamadı ({path}): {exc}")
        return []

    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
    items: List[Tuple[str, dict, str]] = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = {"source": path.name, "page_hint": idx}
        cid = hash_chunk(chunk, path.name, idx)
        items.append((chunk, meta, cid))
    return items


def batched(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def ensure_collection(reset: bool) -> chromadb.api.models.Collection.Collection:
    client = chromadb.PersistentClient(path=str(VECTOR_DIR), settings=Settings(allow_reset=True))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    try:
        return client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    except Exception:
        return client.get_collection(COLLECTION_NAME)


# --------------------------
# Main routine
# --------------------------
def main(
    input_dir: str = "data/",
    reset: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    max_workers: int = DEFAULT_MAX_WORKERS,
    add_batch_size: int = DEFAULT_ADD_BATCH,
) -> None:
    source_dir = Path(input_dir)
    if not source_dir.exists():
        print(f"[WARN] {source_dir} bulunamadı. PDF klasörünü kontrol et.")
        return

    pdf_files = sorted(source_dir.rglob("*.pdf"))
    if not pdf_files:
        print("[WARN] data/ içinde PDF bulunamadı. En az 1 PDF koy.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_batch = DEFAULT_EMBED_BATCH_GPU if device == "cuda" else DEFAULT_EMBED_BATCH_CPU
    print(f"[INFO] Device: {device} | PyMuPDF: {HAS_PYMUPDF} | extract-workers: {max_workers} | embed-batch: {embed_batch}")
    print(f"[INFO] PDF sayısı: {len(pdf_files)} - metin çıkarılıyor...")

    all_chunks: List[Tuple[str, dict, str]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_pdf_chunks, path, chunk_size, overlap): path for path in pdf_files}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                chunks = fut.result()
                all_chunks.extend(chunks)
            except Exception as exc:
                print(f"[WARN] PDF işleme hatası ({path}): {exc}")

    if not all_chunks:
        print("[WARN] Hiç parça çıkmadı.")
        return

    print(f"[INFO] Toplam parça: {len(all_chunks)} - Chroma ile varlık kontrolü yapılıyor...")
    collection = ensure_collection(reset=reset)

    # Skip already ingested ids
    all_ids = [cid for (_, _, cid) in all_chunks]
    existing: set[str] = set()
    for batch in batched(all_ids, 1000):
        try:
            res = collection.get(ids=list(batch))
            if res and res.get("ids"):
                existing.update(res["ids"])
        except Exception:
            pass

    new_items = [(text, meta, cid) for (text, meta, cid) in all_chunks if cid not in existing]
    if not new_items:
        print("[INFO] Yeni parça yok. Bitti.")
        return

    print(f"[INFO] Yeni eklenecek: {len(new_items)} - embedding ve ekleme başlıyor...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    total_added = 0
    for batch in batched(new_items, add_batch_size):
        texts = [text for (text, _, _) in batch]
        metadatas = [meta for (_, meta, _) in batch]
        ids = [cid for (_, _, cid) in batch]

        inputs = [PASSAGE_PREFIX + text for text in texts]
        embeddings = embedder.encode(
            inputs,
            batch_size=embed_batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        total_added += len(batch)
        print(f"   -> Eklendi: {total_added}/{len(new_items)}", end="\r", flush=True)

    print(f"\n[INFO] Tamamlandı. {total_added} parça eklendi. Vektör DB: {VECTOR_DIR} (koleksiyon: {COLLECTION_NAME})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/", help="PDF klasörü (default: data/)")
    parser.add_argument("--reset", action="store_true", help="Koleksiyonu silip baştan kur")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Parça uzunluğu (karakter)")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Parça bindirme (karakter)")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, help="PDF extraction worker sayısı")
    parser.add_argument("--add-batch", type=int, default=DEFAULT_ADD_BATCH, help="Chroma add batch size")
    args = parser.parse_args()

    main(
        input_dir=args.input,
        reset=args.reset,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_workers=args.workers,
        add_batch_size=args.add_batch,
    )
