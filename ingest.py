import os, argparse, uuid
from pathlib import Path
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Ayarlar
DATA_DIR = Path("data")
VECTOR_DIR = Path("vectordb")
COLLECTION_NAME = "docs"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"

# Basit metin parçalama (chunk)
def chunk_text(text: str, chunk_size=600, overlap=150) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def load_pdfs_from_dir(dir_path: Path) -> List[tuple]:
    out = []
    for p in dir_path.rglob("*.pdf"):
        reader = PdfReader(str(p))
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(pages)
        for idx, ch in enumerate(chunk_text(text)):
            meta = {"source": p.name, "page_hint": idx+1}
            out.append((ch, meta))
    return out

def main(input_dir: str):
    # Embedder
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    # Chroma (kalıcı)
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)  # yeniden kurulum kolaylığı
    except Exception:
        pass
    col = client.create_collection(name=COLLECTION_NAME)

    items = load_pdfs_from_dir(Path(input_dir))
    if not items:
        print("⚠️ data/ altında PDF bulunamadı. En az 1 PDF koy.")
        return

    texts = [t for t,_ in items]
    metas = [m for _,m in items]
    ids = [str(uuid.uuid4()) for _ in texts]
    embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()

    col.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
    print(f"✅ {len(texts)} parça eklendi. Vektör DB: {VECTOR_DIR}/ (koleksiyon: {COLLECTION_NAME})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/", help="PDF klasörü")
    args = ap.parse_args()
    main(args.input)
