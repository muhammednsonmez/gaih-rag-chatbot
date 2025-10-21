import os
import io
import glob
import uuid
import time
import streamlit as st
from dotenv import load_dotenv

# RAG hattı (mevcut dosyan)
from rag_pipeline import answer

# PDF & indeksleme için
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from ingest import chunk_text

load_dotenv()

# --------- Sabitler ---------
DATA_DIR = "data"
VECTOR_DIR = "vectordb"
COLLECTION_NAME = "docs"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"

os.makedirs(DATA_DIR, exist_ok=True)

# --------- Yardımcılar ---------
def load_css(path: str):
    """Tek dosyadan CSS yükle."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS bulunamadı: {path}")

def list_pdfs():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    return [os.path.basename(p) for p in files]

def save_bytes_to_data(original_name: str, bytes_data: bytes) -> str:
    """Yüklenen PDF'i data/ altına KULLANICI TETİĞİYLE kaydeder. Aynı isim varsa (1),(2)... ekler."""
    base = os.path.basename(original_name)
    name, ext = os.path.splitext(base)
    target = os.path.join(DATA_DIR, base)
    c = 1
    while os.path.exists(target):
        target = os.path.join(DATA_DIR, f"{name} ({c}){ext}")
        c += 1
    with open(target, "wb") as f:
        f.write(bytes_data)
    return target

def index_pdf_file(pdf_path: str, delete_existing: bool = True):
    """Tek PDF'i okuyup vektör indekse ekler. Aynı kaynaktan varsa önce siler (dup. önler)."""
    # 1) PDF metni
    reader = PdfReader(pdf_path)
    text = "\n".join([(p.extract_text() or "") for p in reader.pages])
    chunks = chunk_text(text)
    if not chunks:
        raise RuntimeError("Metin çıkarılamadı (taranmış PDF olabilir).")

    # 2) Chroma koleksiyon
    client = chromadb.PersistentClient(path=VECTOR_DIR)
    col = client.get_collection(COLLECTION_NAME)

    # Aynı source varsa sil
    if delete_existing:
        try:
            col.delete(where={"source": os.path.basename(pdf_path)})
        except Exception:
            pass

    # 3) Embedding + ekleme
    emb = SentenceTransformer(EMBED_MODEL_NAME)
    vecs = emb.encode(chunks, normalize_embeddings=True).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metas = [{"source": os.path.basename(pdf_path), "page_hint": i + 1} for i in range(len(chunks))]
    col.add(ids=ids, documents=chunks, metadatas=metas, embeddings=vecs)

def rewrite_to_english(q: str) -> str:
    """Gemini ile kısa İngilizce yeniden yazım. Hata olursa orijinali döndürür."""
    import requests, re
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
st.set_page_config(page_title="TR-FAQ RAG Chatbot", page_icon="🔎", layout="centered")
load_css("assets/styles.css")

# Hero header
st.markdown(
    """
    <div class="hero-wrap">
      <span class="hero-icon">🤖</span>
      <span class="hero-title">TR-FAQ RAG Chatbot</span>
    </div>
    <div class="hero-sub">
      PDF belgelerinden akıllı yanıt veren çok dilli RAG uygulaması<br>
      <b>Model:</b> Gemini • <b>Vektör DB:</b> Chroma • <b>Türkçe öncelikli</b>
    </div>
    """,
    unsafe_allow_html=True
)

# --------- Sidebar ---------
with st.sidebar:
    st.subheader("Ayarlar")

    providers = ["gemini", "openai"]
    default_provider = "gemini"
    if os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        default_provider = "openai"
    if "provider" not in st.session_state:
        st.session_state.provider = default_provider

    st.session_state.provider = st.radio(
        "LLM sağlayıcı",
        providers,
        index=providers.index(st.session_state.provider),
        horizontal=False,
    )
    top_k = st.slider("Top K", 1, 8, 4)

    # Çok dilli (çeviri) seçeneği
    if "multilingual" not in st.session_state:
        st.session_state.multilingual = False
    st.session_state.multilingual = st.toggle(
        "Çok dilli (çeviri ile)",
        value=st.session_state.multilingual,
        help="Açıksa, TR dışındaki dilleri İngilizceye yeniden yazarak aramada kullanır; yanıt Türkçe olur."
    )

    st.markdown("---")
    st.markdown("### data/ içindeki PDF'ler")
    pdfs = list_pdfs()
    if pdfs:
        for fn in pdfs:
            st.write("• ", fn)
    else:
        st.caption("data/ klasöründe PDF yok.")

    # uploader'ı resetlemek için döngüsel key kullan
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded = st.file_uploader(
        "PDF seç", type=["pdf"],
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded:
        size_mb = len(uploaded.getvalue()) / (1024 * 1024)
        st.caption(f"Seçildi: {uploaded.name} • {size_mb:.2f} MB")

        if st.button("Kaydet & indeksle", use_container_width=True, key=f"btn_ingest_{st.session_state.uploader_key}"):
            try:
                bytes_data = uploaded.getvalue()
                target = save_bytes_to_data(uploaded.name, bytes_data)
                with st.spinner(f"İndeksleniyor: {os.path.basename(target)}"):
                    t0 = time.time()
                    index_pdf_file(target, delete_existing=True)  # aynı kaynak varsa sil
                    dt = time.time() - t0
                st.success(f"{os.path.basename(target)} eklendi ve indekslendi ({dt:.2f} sn).")
                # uploader'ı sıfırla
                st.session_state.uploader_key += 1
                st.rerun()
            except Exception as e:
                st.error(f"Yükleme/indeksleme hatası: {e}")

# --------- Session State ---------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "cancel" not in st.session_state:
    st.session_state.cancel = False
if "running" not in st.session_state:
    st.session_state.running = False

# --------- Sorgu Formu ---------
with st.form("qa_form", clear_on_submit=True):
    q = st.text_input("Sorunu yaz:", placeholder="Örn: Category ID 171146 nedir?")
    c1, c2 = st.columns([1, 1], gap="small")
    with c1:
        send = st.form_submit_button("Gönder", use_container_width=True, type="primary")
    with c2:
        toggle_cancel = st.form_submit_button("İptal (aç/kapat)", use_container_width=True)
    st.caption("Not: Formu göndermeden sayfadaki diğer butonlar yeni sorgu başlatmaz.")

# İptal toggle (aç/kapat)
if toggle_cancel:
    st.session_state.cancel = not st.session_state.cancel
    st.markdown(
        f"<div style='text-align:right; font-weight:500; color:#ccc;'>"
        f"İptal modu: {'🟥 AÇIK — yeni sorgular başlatılmaz.' if st.session_state.cancel else '🟩 KAPALI — sorgular çalışır.'}"
        f"</div>",
        unsafe_allow_html=True
    )

# --------- Sorgu Çalıştırma ---------
if send:
    if st.session_state.cancel:
        st.info("İptal açık. Sorgu başlatılmadı.")
    elif not q.strip():
        st.warning("Soru boş olamaz.")
    elif st.session_state.running:
        st.info("Zaten bir sorgu çalışıyor.")
    else:
        try:
            st.session_state.running = True

            # Çok dilli mod: EN reformülasyonu retrieval ipucu olarak eklensin
            combined_q = q
            if st.session_state.multilingual:
                q_en = rewrite_to_english(q)
                if q_en and q_en.strip().lower() != q.strip().lower():
                    combined_q = f"{q}\n\n(English reformulation for retrieval: {q_en})"

            t0 = time.time()
            with st.spinner("Yanıt hazırlanıyor…"):
                out = answer(combined_q, top_k=top_k, provider=st.session_state.provider)
            dt = time.time() - t0
            st.caption(f"⏱️ Sorgu süresi: **{dt:.2f} saniye**")

            st.session_state.history.append(("user", q))
            st.session_state.history.append(("assistant", out["answer"]))
            st.session_state.last_sources = out.get("sources", [])
        except Exception as e:
            st.error(f"Model çağrısı/indeks hatası: {e}")
        finally:
            st.session_state.running = False

# --------- Sohbet Geçmişi ---------
ICON_USER = "🧠"       # kullanıcı avatarı
ICON_ASSISTANT = "⚙️"  # asistan avatarı

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
