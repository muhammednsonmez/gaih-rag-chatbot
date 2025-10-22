# debug_retrieval.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
QUERY_PREFIX = "query: "

VECTOR_DIR = "vectordb"
COLLECTION_NAME = "docs"

client = chromadb.PersistentClient(path=VECTOR_DIR, settings=Settings(allow_reset=True))
col = client.get_collection(COLLECTION_NAME)
_embedder = SentenceTransformer(EMBED_MODEL_NAME)


def debug_query(q, n=5):
    q_emb = _embedder.encode([QUERY_PREFIX + q], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=n, include=["documents", "metadatas", "distances"])
    print("\n=== Sorgu:", q, "===")
    if not res or not res.get("ids") or not res["ids"] or len(res["ids"][0]) == 0:
        print("-> Koleksiyondan sonuç gelmedi.")
        return
    for i, (did, doc, meta, dist) in enumerate(
        zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
    ):
        sim = max(0.0, 1.0 - float(dist))
        print(f"\n#{i} id={did} dist={dist:.4f} sim={sim:.4f}")
        print("meta:", meta)
        print("doküman (ilk 400):", (doc or "")[:400].replace("\n", " "))


if __name__ == "__main__":
    debug_query("Kali Linux güncelleme nasıl yapılır?")
    debug_query("phishing nedir")
    debug_query("Kali'de ağ arayüzlerini listeleme komutu nedir?")
    debug_query("Route table from msfconsole nedir")
