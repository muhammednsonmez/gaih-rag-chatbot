import os
import re
import hashlib
from typing import List, Dict
from functools import lru_cache
import chromadb
import requests
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
VECTOR_DIR = "vectordb"
COLLECTION_NAME = "docs"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"

# -----------------------------
# Embedding (lazy global)
# -----------------------------
@lru_cache(maxsize=1)
def _get_embedder():
    """Load embedding model lazily to avoid blocking app startup."""
    return SentenceTransformer(EMBED_MODEL_NAME)

# -----------------------------
# Chroma helpers
# -----------------------------
def _client_collection():
    client = chromadb.PersistentClient(path=VECTOR_DIR)
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        raise RuntimeError(
            "Vektör indeksi bulunamadı. Lütfen önce ingest çalıştırın:\n"
            "  python ingest.py --input data/\n"
            "Ayrıca data klasöründe en az 1 PDF olduğundan emin olun."
        )

# -----------------------------
# Basit sayfa sayfa okuma (RAM dostu)
# -----------------------------
@lru_cache(maxsize=1)
def _all_docs_in_ram():
    col = _client_collection()
    total = col.count()
    off, docs, metas = 0, [], []
    while off < total:
        got = col.get(include=["documents", "metadatas"], limit=500, offset=off)
        docs.extend(got.get("documents", []) or [])
        metas.extend(got.get("metadatas", []) or [])
        off += 500
    return docs, metas

# -----------------------------
# Keyword / number scan
# -----------------------------
def _keyword_hits(col, query: str, max_hits: int = 200):
    nums = re.findall(r"\d{3,}", query)
    words = [w.lower() for w in re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", query)]
    if not nums and not words:
        return []

    hits, seen = [], set()
    docs, metas = _all_docs_in_ram()
    for doc, meta in zip(docs, metas):
        dl = doc.lower()
        has_all_nums = all(n in doc for n in nums) if nums else True
        has_some_words = any(w in dl for w in words) if words else True
        if has_all_nums and has_some_words:
            score = 0.0
            for n in nums:
                score += doc.count(n) * 2.0
            for w in set(words):
                score += dl.count(w) * 1.0
            key = (meta.get("source", "?"), meta.get("page_hint", "?"), hash(doc))
            if key not in seen:
                seen.add(key)
                hits.append({"text": doc, "meta": meta, "score_kw": float(score)})
        if len(hits) >= max_hits:
            break

    hits.sort(key=lambda x: x["score_kw"], reverse=True)
    return hits[:max_hits]

# -----------------------------
# Hybrid retrieve
# -----------------------------
def retrieve(query: str, top_k: int = 4) -> List[Dict]:
    col = _client_collection()

    only_numbers = re.fullmatch(r"\D*\d{3,}\D*", query.strip()) is not None
    kw_docs = _keyword_hits(col, query, max_hits=200)

    if only_numbers and kw_docs:
        merged = kw_docs[:max(top_k, 6)]
        max_kw = max(d["score_kw"] for d in merged) or 1.0
        for d in merged:
            d["score"] = 0.4 * (d["score_kw"] / max_kw)
        return merged[:top_k]

    embedder = _get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    vres = col.query(
        query_embeddings=q_emb,
        n_results=max(top_k, 12),
        include=["documents", "metadatas", "distances"]
    )
    vector_docs = []
    for doc, meta, dist in zip(vres["documents"][0], vres["metadatas"][0], vres["distances"][0]):
        sim = 1.0 / (1.0 + float(dist))
        vector_docs.append({"text": doc, "meta": meta, "score_vec": sim})

    bag = {}
    for d in vector_docs:
        h = hashlib.md5(d["text"].encode("utf-8")).hexdigest()
        bag[h] = {"text": d["text"], "meta": d["meta"], "score_vec": d["score_vec"], "score_kw": 0.0}
    for d in kw_docs:
        h = hashlib.md5(d["text"].encode("utf-8")).hexdigest()
        if h in bag:
            bag[h]["score_kw"] = max(bag[h]["score_kw"], d.get("score_kw", 0.0))
        else:
            bag[h] = {"text": d["text"], "meta": d["meta"], "score_vec": 0.0, "score_kw": d.get("score_kw", 0.0)}

    merged = list(bag.values())
    for d in merged:
        d["score"] = 0.6 * d["score_vec"] + 0.4 * (d["score_kw"] / 10.0)

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]

# -----------------------------
# Gemini caller (REST)
# -----------------------------
def _generate_gemini(system_prompt: str, user_prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env değişkeni yok.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    payload = {"contents": [{"parts": [{"text": system_prompt + "\n\n" + user_prompt}]}]}

    # v1 ve v1beta varyasyonlarını dene
    combos = [("v1", False), ("v1", True), ("v1beta", False), ("v1beta", True)]
    last_info = None
    for api_ver, add_prefix in combos:
        name = f"models/{model}" if add_prefix else model
        url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{name}:generateContent?key={api_key}"
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 404:
                last_info = f"{api_ver}:{name} -> 404"
                continue
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            last_info = f"{api_ver}:{name} -> {e}"
            continue

    try:
        names = []
        for api_ver in ("v1", "v1beta"):
            list_url = f"https://generativelanguage.googleapis.com/{api_ver}/models?key={api_key}"
            resp = requests.get(list_url, timeout=15)
            if resp.ok:
                models = resp.json().get("models", [])
                names += [m.get("name", "") for m in models]
        raise RuntimeError(f"Gemini çağrısı başarısız. Denemeler: {last_info}. Sunucunun bildirdiği örnek modeller: {names[:10]}")
    except Exception as e:
        raise RuntimeError(f"Gemini çağrısı başarısız. Son deneme: {last_info}. Hata: {e}")

# -----------------------------
# Public API (yalnızca Gemini)
# -----------------------------
def answer(query: str, top_k: int = 4, retrieval_query: str | None = None) -> Dict:
    rq = retrieval_query or query
    ctx_docs = retrieve(rq, top_k=top_k)

    context = "\n\n".join(
        [f"[Kaynak {i+1}] {d['meta'].get('source','?')} (parça {d['meta'].get('page_hint','?')})\n{d['text']}"
         for i, d in enumerate(ctx_docs)]
    )
    system_prompt = (
        "Sen Türkçe konuşan bir yardımcı botsun. Öncelikle verilen bağlamı kullan; "
        "bağlam yetersizse 'Bu konuda elimde yeterli bilgi yok.' de. "
        "Cevabın sonunda kullandığın kaynak numaralarını köşeli parantezle göster."
    )
    user_prompt = f"Soru: {query}\n\nBağlam:\n{context}"

    text = _generate_gemini(system_prompt, user_prompt)
    return {"answer": text, "sources": ctx_docs}
