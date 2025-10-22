# app_streamlit.py  — Sadece OKUMA (no-ingest) Streamlit uygulaması
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# Proje içi
from rag_pipeline import answer  # answer(q, top_k) -> {"answer": str, "sources": [...]}

load_dotenv()

# ----------------- Ayarlar -----------------
VECTOR_DIR = os.getenv("RAG_VECTOR_DIR", "vectordb").strip()
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "docs").strip()

# Chroma telemetry ayarı: mevcut indeksle uyuşmalı (true/false)
# (Lokalde hangi değerde oluşturduysan sunucuda da aynı olsun.)
ANON_TELEMETRY = os.getenv("ANONYMIZED_TELEMETRY", "true").lower() == "true"

# ----------------- UI -----------------
st.set_page_config(page_title="Kali Linux RAG (Read-Only)", page_icon="🔎", layout="centered")
st.markdown(
    """
    <div style="text-align:center;margin-top:10px">
      <h2>🤖 Kali Linux RAG (Read-Only)</h2>
      <div style="opacity:.8">Model: <b>Gemini</b> • Vektör DB: <b>Chroma</b></div>
    </div>
    <hr style="opacity:.1;margin:16px 0 8px 0;">
    """,
    unsafe_allow_html=True
)

# ----------------- Chroma (CACHE) -----------------
@st.cache_resource(show_spinner=False)
def get_chroma_client():
    # Not: Burada ingest YOK. Sadece mevcut klasörü okuyoruz.
    if not Path(VECTOR_DIR).exists():
        raise FileNotFoundError(
            f"Vektör dizini bulunamadı: {VECTOR_DIR}\n"
            f"Önce lokalde ingest yap ve 'vectordb/' klasörünü sunucuya kopyala."
        )
    settings = Settings(anonymized_telemetry=ANON_TELEMETRY)
    return chromadb.PersistentClient(path=VECTOR_DIR, settings=settings)

@st.cache_resource(show_spinner=False)
def get_collection():
    client = get_chroma_client()
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        # Bazı ortamlarda get_or_create yerine get tercih ederiz;
        # create etmek bile istemiyoruz, çünkü ingest app dışında.
        raise RuntimeError(
            f"'{COLLECTION_NAME}' koleksiyonu bulunamadı. "
            f"İndeksi lokalde oluşturup klasörü aynen kopyaladığından emin ol."
        )
    return col

def collection_count(col) -> int:
    try:
        return int(col.count())
    except Exception as e:
        raise RuntimeError(f"Koleksiyon sayımı başarısız: {e}")

# ----------------- İndeks Sağlığı -----------------
try:
    with st.spinner("Vektör indeksi kontrol ediliyor…"):
        col = get_collection()
        total = collection_count(col)
        if total == 0:
            st.error(
                "Koleksiyon boş görünüyor. Bu uygulama ingest yapmaz.\n"
                "👉 Lokalde `python ingest.py` çalıştır, oluşan `vectordb/` klasörünü sunucuya kopyala."
            )
            st.stop()
except Exception as e:
    st.error(f"İndeks erişim hatası: {e}")
    st.stop()

# ----------------- Sidebar -----------------
with st.sidebar:
    st.subheader("Ayarlar")
    top_k = st.slider("Top K", 1, 10, 4)
    st.caption(f"Koleksiyon: {COLLECTION_NAME} • Parça sayısı: {collection_count(col)}")

    # Kaynak özetini hafifçe göster (büyük koleksiyonlarda sayfalı okuma yapma)
    try:
        got = col.get(include=["metadatas"], limit=min(1000, collection_count(col)))
        metas = got.get("metadatas") or []
        src_map = {}
        for m in metas:
            if not m:
                continue
            src = m.get("source") or m.get("file") or "?"
            src_map[src] = src_map.get(src, 0) + 1
        if src_map:
            st.markdown("**Örnek Kaynaklar:**")
            for s_name, cnt in list(sorted(src_map.items(), key=lambda x: (-x[1], x[0])))[:10]:
                st.write(f"• {s_name} _(parça: {cnt})_")
        else:
            st.caption("Kaynak metadatası okunamadı.")
    except Exception as e:
        st.caption(f"Kaynak listesi hatası: {e}")

# ----------------- Soru Formu -----------------
with st.form("qa_form", clear_on_submit=False):
    q = st.text_input("Sorunu yaz:", placeholder="Örn: Kali Linux nedir?")
    send = st.form_submit_button("Gönder", use_container_width=True, type="primary")

if send:
    if not q.strip():
        st.warning("Soru boş olamaz.")
    else:
        t0 = time.time()
        try:
            with st.spinner("Yanıt hazırlanıyor…"):
                out = answer(q, top_k=top_k)  # RAG pipeline sadece OKUMA modunda çalışmalı
            dt = time.time() - t0

            st.success("Yanıt")
            st.markdown(out.get("answer", ""))
            st.caption(f"⏱️ Sorgu süresi: {dt:.2f}s")

            sources = out.get("sources", []) or []
            if sources:
                st.markdown("### Kaynaklar")
                for i, s in enumerate(sources, 1):
                    meta = s.get("meta", {}) if isinstance(s, dict) else {}
                    src = meta.get("source") or meta.get("file") or "?"
                    page_hint = meta.get("page_hint") or meta.get("page") or meta.get("chunk") or "?"
                    with st.expander(f"[{i}] {src} (parça {page_hint})"):
                        st.write((s.get("text", "") if isinstance(s, dict) else str(s))[:1500])
        except Exception as e:
            st.error(f"Model/indeks çağrısı hatası: {e}")
