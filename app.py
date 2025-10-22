# app.py
import os
import uuid
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# 3rd party
import chromadb

# Proje içi
from ingest import main as ingest_main   # ingest_main("data/") bekleniyor
from rag_pipeline import answer          # answer(q, top_k) -> {"answer": str, "sources": [...]}

load_dotenv()

# ----------------- Sabitler -----------------
VECTOR_DIR = os.getenv("RAG_VECTOR_DIR", "vectordb").strip()
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "docs").strip()
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "data")).resolve()

# ----------------- CSS -----------------
def load_css(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# ----------------- Chroma Yardımcıları -----------------
@st.cache_resource(show_spinner=False)
def get_chroma_client():
    # Telemetry sessiz
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    return chromadb.PersistentClient(path=VECTOR_DIR)

@st.cache_resource(show_spinner=False)
def get_or_create_collection():
    client = get_chroma_client()
    # HNSW metrik belirtmek (cosine) arama kalitesi için iyi bir varsayılan
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        col = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return col

def collection_count(col) -> int:
    try:
        return int(col.count())
    except Exception:
        return 0

def ensure_chroma_index():
    """
    1) Koleksiyon var mı? Yoksa oluştur.
    2) Boş mu? Boşsa ingest dene (yalnızca DATA_DIR içinde PDF varsa).
       -> Bu akış "lokalde indexle sonra sunucuya at" stratejini destekler:
          Sunucuda koleksiyon zaten doluysa ingest'e girmez.
    """
    col = get_or_create_collection()
    if collection_count(col) > 0:
        return  # indeks hazır

    # Koleksiyon boş → ingest gerekebilir
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(
            f"İndeks boş görünüyor ve {DATA_DIR} içinde PDF bulunamadı. "
            f"Önce lokalde ingest edip sonra sunucuya yükle, ya da PDF'leri {DATA_DIR}/ altına koy."
        )

    # Ingest çalıştır
    ingest_main(str(DATA_DIR))

    # Persist edilen veriyi görmek için koleksiyonu tekrar al
    # (bazı ortamlarda ingest ayrı process/client ile yazmış olabilir)
    time.sleep(0.2)
    col = get_or_create_collection()

    if collection_count(col) == 0:
        raise RuntimeError("Ingest bitti ama koleksiyon hâlâ boş görünüyor. Ingest tarafındaki ayarları (persist path/collection) kontrol et.")

# ----------------- Dil Yardımcısı (opsiyonel) -----------------
def rewrite_to_english(q: str) -> str:
    """
    Gerekirse kısa İngilizce yeniden yazım. GEMINI_API_KEY yoksa dokunmaz.
    Sadece TR karakteri veya uzun rakam içeren sorularda dener.
    """
    import re
    need = bool(re.search(r"[çğıöşüÇĞİÖŞÜ]", q)) or bool(re.search(r"\d{3,}", q))
    if not need:
        return q

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return q

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
    prompt = (
        "Rewrite the user question in concise English for better document retrieval. "
        "Return only the rewritten query without extra text."
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}, {"parts": [{"text": q}]}]}

    try:
        import requests
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return q

# ----------------- UI Başlangıç -----------------
st.set_page_config(page_title="Kali Linux Multilingual-Turkish RAG Chatbot", page_icon="🔎", layout="centered")
load_css("assets/styles.css")

st.markdown(
    """
    <div class="hero-wrap" style="text-align:center;margin-top:14px;">
      <span class="hero-icon">🤖</span>
      <span class="hero-title">Kali Linux Multilingual-Turkish RAG Chatbot</span>
    </div>
    <div class="hero-sub" style="text-align:center;opacity:.9;">
      Model: <b>Gemini</b> • Vektör DB: <b>Chroma</b>
    </div>
    <hr style="opacity:.1;margin:18px 0 6px 0;">
    """,
    unsafe_allow_html=True
)

# ----------------- Session State -----------------
ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("last_sources", [])
ss.setdefault("running", False)
ss.setdefault("run_token", None)
ss.setdefault("cancel_requested", False)
ss.setdefault("multilingual", False)

# ----------------- İndeks Hazırlığı -----------------
try:
    with st.spinner("İndeks kontrol ediliyor…"):
        ensure_chroma_index()
except Exception as e:
    st.error(f"Vektör indeksi bulunamadı/hazırlanamadı. Detay: {e}")
    st.stop()

# ----------------- Sidebar -----------------
with st.sidebar:
    st.subheader("Ayarlar")
    top_k = st.slider("Top K", 1, 8, 4)
    ss.multilingual = st.toggle(
        "Çok dilli (çeviri ile)",
        value=ss.multilingual,
        help="Açıksa, TR dışındaki dilleri İngilizceye yeniden yazar; yanıt Türkçe olur."
    )
    st.markdown("---")
    st.markdown("**İndeks Özeti (Chroma)**")

    try:
        col = get_or_create_collection()
        total = collection_count(col)
        st.caption(f"Koleksiyon: {COLLECTION_NAME} • Parça sayısı: {total}")

        # Kaynak istatistikleri
        sources = {}
        offset, step = 0, 500
        while offset < total:
            got = col.get(include=["metadatas"], limit=step, offset=offset)
            metas = got.get("metadatas") or []
            for m in metas:
                if not m:
                    continue
                src = m.get("source") or m.get("file") or "?"
                sources[src] = sources.get(src, 0) + 1
            offset += step

        if sources:
            st.markdown("**Kaynaklar (indeksten):**")
            for s_name, cnt in sorted(sources.items(), key=lambda x: x[0].lower()):
                st.write(f"• {s_name} _(parça: {cnt})_")
        else:
            st.caption("İndekste kaynak meta bulunamadı.")
    except Exception as e:
        st.caption(f"Chroma erişim hatası: {e}")

# ----------------- Form -----------------
with st.form("qa_form", clear_on_submit=True):
    q = st.text_input("Sorunu yaz:", placeholder="Örn: Kali Linux nedir?")
    c1, c2 = st.columns([1, 1], gap="small")
    with c1:
        send = st.form_submit_button("Gönder", use_container_width=True, type="primary")
    with c2:
        cancel_now = st.form_submit_button("İptal Et", use_container_width=True)
    st.caption("Not: Gönder'e bastıktan sonra 'İptal Et' ile o anki sorguyu iptal edebilirsin.")

# Tek seferlik iptal
if cancel_now:
    if ss.get("running", False):
        ss.cancel_requested = True
        ss.run_token = None
        ss.running = False
        st.info("Sorgu iptal edildi.")
        st.rerun()

# ----------------- Sorgu Çalıştırma -----------------
if send:
    if not q.strip():
        st.warning("Soru boş olamaz.")
    else:
        my_token = uuid.uuid4().hex
        ss.run_token = my_token
        ss.cancel_requested = False

        try:
            ss.running = True
            combined_q = q
            if ss.multilingual:
                q_en = rewrite_to_english(q)
                if q_en and q_en.strip().lower() != q.strip().lower():
                    combined_q = f"{q}\n\n(English reformulation for retrieval: {q_en})"

            t0 = time.time()
            with st.spinner("Yanıt hazırlanıyor…"):
                out = answer(combined_q, top_k=top_k)  # {"answer": str, "sources": [...]}
            dt = time.time() - t0

            # İptal kontrolü (geç gelen yanıtı bastırma)
            if ss.run_token != my_token or ss.cancel_requested:
                ss.cancel_requested = False
                st.stop()

            # Cevabı yaz
            final_answer = out.get("answer", "")
            ss.history.append(("user", q))
            ss.history.append(("assistant", final_answer))
            ss.last_sources = out.get("sources", []) or []

            st.caption(f"⏱️ Sorgu süresi: **{dt:.2f} saniye**")

        except Exception as e:
            st.error(f"Model/indeks çağrısı hatası: {e}")
        finally:
            ss.running = False

# ----------------- Sohbet Geçmişi -----------------
ICON_USER = "🧠"
ICON_ASSISTANT = "⚙️"
for role, msg in ss.history:
    if role == "user":
        with st.chat_message("user", avatar=ICON_USER):
            st.markdown(f"**Kullanıcı:** {msg}")
    else:
        with st.chat_message("assistant", avatar=ICON_ASSISTANT):
            st.markdown(f"**Asistan:** {msg}")

# ----------------- Kaynaklar -----------------
if ss.last_sources:
    st.markdown("### Kaynaklar")
    for i, s in enumerate(ss.last_sources, 1):
        meta = s.get("meta", {}) if isinstance(s, dict) else {}
        # Esnek başlık
        src = meta.get("source") or meta.get("file") or "?"
        page_hint = meta.get("page_hint") or meta.get("page") or meta.get("chunk") or "?"
        title = f"[{i}] {src} (parça {page_hint})"
        with st.expander(title):
            text_preview = s.get("text", "")[:1500] if isinstance(s, dict) else str(s)[:1500]
            st.write(text_preview)

# ----------------- Sohbet Dışa Aktar -----------------
if ss.history:
    export_text = ""
    for role, msg in ss.history:
        prefix = "Kullanıcı" if role == "user" else "Asistan"
        export_text += f"{prefix}: {msg}\n\n"
    st.download_button(
        "Sohbeti .txt olarak indir",
        data=export_text.encode("utf-8"),
        file_name="chat_export.txt",
        mime="text/plain",
        use_container_width=True,
    )
