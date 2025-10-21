import os
import uuid
import time
from ingest import main as ingest_main
import streamlit as st
from dotenv import load_dotenv
import chromadb
from rag_pipeline import answer

load_dotenv()

# --------- Sabitler ---------
VECTOR_DIR = "vectordb"
COLLECTION_NAME = "docs"

# --------- Yardımcılar ---------
def load_css(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def ensure_chroma_index():
    import chromadb
    client = chromadb.PersistentClient(path=VECTOR_DIR)
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        # koleksiyon yoksa data/ klasöründen indeksle
        ingest_main("data/")

def rewrite_to_english(q: str) -> str:
    """Gemini'ye ipucu için kısa İngilizce yeniden yazım. Hata olursa orijinali döndürür."""
    import re, requests
    need = bool(re.search(r"[çğıöşüÇĞİÖŞÜ]", q)) or bool(re.search(r"\d{3,}", q))
    if not need:
        return q
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return q
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
    prompt = ("Rewrite the user question in concise English for better document retrieval. "
              "Return only the rewritten query without extra text.")
    payload = {"contents": [{"parts": [{"text": prompt}]}, {"parts": [{"text": q}]}]}
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return q

# --------- UI Başlangıç ---------
st.set_page_config(page_title="Kali Linux Multilingual-Turkish RAG Chatbot", page_icon="🔎", layout="centered")
load_css("assets/styles.css")
ensure_chroma_index()

# Header
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

# --------- Session State ---------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "running" not in st.session_state:
    st.session_state.running = False
if "run_token" not in st.session_state:
    st.session_state.run_token = None
if "cancel_requested" not in st.session_state:
    st.session_state.cancel_requested = False

# --------- Sidebar ---------
with st.sidebar:
    st.subheader("Ayarlar")
    top_k = st.slider("Top K", 1, 8, 4)

    if "multilingual" not in st.session_state:
        st.session_state.multilingual = False
    st.session_state.multilingual = st.toggle(
        "Çok dilli (çeviri ile)",
        value=st.session_state.multilingual,
        help="Açıksa, TR dışındaki dilleri İngilizceye yeniden yazar; yanıt Türkçe olur."
    )

    st.markdown("---")
    st.markdown("**İndeks Özeti (Chroma)**")
    try:
        client = chromadb.PersistentClient(path=VECTOR_DIR)
        col = client.get_collection(COLLECTION_NAME)
        total = col.count()
        st.caption(f"Koleksiyon: {COLLECTION_NAME} • Parça sayısı: {total}")

        # Tüm kaynakları sayfa sayfa oku
        sources = {}
        offset, step = 0, 500
        while offset < total:
            got = col.get(include=["metadatas"], limit=step, offset=offset)
            metas = got.get("metadatas", []) or []
            for m in metas:
                if not m:
                    continue
                src = m.get("source", "?")
                sources[src] = sources.get(src, 0) + 1
            offset += step

        if sources:
            st.markdown("**Kaynaklar (indeksten):**")
            for s, cnt in sorted(sources.items(), key=lambda x: x[0].lower()):
                st.write(f"• {s}  _(parça: {cnt})_")
        else:
            st.caption("İndekste kaynak meta bulunamadı.")
    except Exception as e:
        st.caption(f"Chroma erişim hatası: {e}")

# --------- Form ---------
with st.form("qa_form", clear_on_submit=True):
    q = st.text_input("Sorunu yaz:", placeholder="Örn: Kali Linux nedir?")
    c1, c2 = st.columns([1, 1], gap="small")
    with c1:
        send = st.form_submit_button("Gönder", use_container_width=True, type="primary")
    with c2:
        cancel_now = st.form_submit_button("İptal Et", use_container_width=True)
    st.caption("Not: Gönder'e bastıktan sonra 'İptal Et' ile o anki sorguyu iptal edebilirsin.")

# Tek seferlik iptal
if 'cancel_now' in locals() and cancel_now:
    if st.session_state.get("running", False):
        st.session_state.cancel_requested = True
        st.session_state.run_token = None
        st.session_state.running = False
    st.info("Sorgu iptal edildi.")
    st.rerun()

# --------- Sorgu Çalıştırma ---------
if send:
    if not q.strip():
        st.warning("Soru boş olamaz.")
    else:
        my_token = uuid.uuid4().hex
        st.session_state.run_token = my_token
        if st.session_state.running:
            st.session_state.running = False

        try:
            st.session_state.running = True

            combined_q = q
            if st.session_state.multilingual:
                q_en = rewrite_to_english(q)
                if q_en and q_en.strip().lower() != q.strip().lower():
                    combined_q = f"{q}\n\n(English reformulation for retrieval: {q_en})"

            t0 = time.time()
            with st.spinner("Yanıt hazırlanıyor…"):
                out = answer(combined_q, top_k=top_k)  # yalnızca Gemini
            dt = time.time() - t0

            # İptal kontrolü — geç yanıtı yazma
            if st.session_state.run_token != my_token or st.session_state.cancel_requested:
                st.session_state.cancel_requested = False
                st.stop()

            st.caption(f"⏱️ Sorgu süresi: **{dt:.2f} saniye**")
            st.session_state.history.append(("user", q))
            st.session_state.history.append(("assistant", out["answer"]))
            st.session_state.last_sources = out.get("sources", [])
        except Exception as e:
            st.error(f"Model çağrısı/indeks hatası: {e}")
        finally:
            st.session_state.running = False

# --------- Sohbet Geçmişi ---------
ICON_USER = "🧠"
ICON_ASSISTANT = "⚙️"
for role, msg in st.session_state.history:
    if role == "user":
        with st.chat_message("user", avatar=ICON_USER):
            st.markdown(f"**Kullanıcı:** {msg}")
    else:
        with st.chat_message("assistant", avatar=ICON_ASSISTANT):
            st.markdown(f"**Asistan:** {msg}")

# --------- Kaynaklar ---------
if st.session_state.last_sources:
    st.markdown("### Kaynaklar")
    for i, s in enumerate(st.session_state.last_sources, 1):
        meta = s.get("meta", {})
        title = f"[{i}] {meta.get('source','?')} (parça {meta.get('page_hint','?')})"
        with st.expander(title):
            st.write(s.get("text", "")[:1500])

# --------- Sohbeti dışa aktar ---------
if st.session_state.history:
    export_text = ""
    for role, msg in st.session_state.history:
        prefix = "Kullanıcı" if role == "user" else "Asistan"
        export_text += f"{prefix}: {msg}\n\n"
    st.download_button(
        "Sohbeti .txt olarak indir",
        data=export_text.encode("utf-8"),
        file_name="chat_export.txt",
        mime="text/plain",
        use_container_width=True,
    )
