# TR-FAQ RAG Chatbot

Gemini destekli **Retrieval-Augmented Generation** (RAG) chatbotu. `data/` klasörüne eklediğiniz PDF dosyalarını indeksleyerek Streamlit arayüzü üzerinden kaynak gösteren yanıtlar üretir. Uygulama varsayılan olarak Türkçe konuşur ve isteğe bağlı olarak çok dilli sorguları İngilizce’ye yeniden yazarak daha isabetli sonuçlar döndürür.

---

## Özellikler
- **Gemini-only**: Tüm yanıtlar Google Gemini API üzerinden üretilir (OpenAI bağımlılığı yoktur).
- **Kalıcı vektör veritabanı**: ChromaDB ile hibrit (vektör + anahtar kelime) arama.
- **Çok dilli sorgu modu**: Türkçe dışındaki soruları otomatik çevirerek bağlam toplamayı iyileştirir.
- **Streamlit arayüzü**: Top-K seçimi, PDF listesi, sohbet geçmişi ve kaynak gösterimi.
- **PDF yükleme & indeksleme**: PDF’leri uygulama içinden veya komut satırından ekleyebilirsiniz.
- **İptal & sohbet dışa aktarma**: Uzayan sorguları iptal edin, konuşmayı `.txt` olarak indirin.

---

## Başlangıç
### Gereksinimler
- Python 3.10+
- Google Gemini API anahtarı (Google AI Studio)
- `pip`, `virtualenv` (opsiyonel ama tavsiye edilir)

### Kurulum Adımları
1. **Depoyu içeri alın**
   ```bash
   git clone <repo-url>
   cd gaih-rag-chatbot
   ```
2. **Sanal ortam (önerilir)**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```
3. **Bağımlılıkları yükleyin**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **.env dosyasını oluşturun**
   `.env` dosyası örneği:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   GEMINI_MODEL=gemini-2.5-flash   # opsiyonel, varsayılan bu model
   ```

> ⚠️ Uygulama yalnızca Gemini ile çalışır. `GEMINI_API_KEY` olmadan başlatamazsınız.

---

## PDF’leri İndeksleme
Uygulama ilk açıldığında `vectordb/` içinde koleksiyon bulamazsa hata verir. PDF’leri indekslemek için:
```bash
python ingest.py --input data/
```

- `data/` klasörüne koyduğunuz her PDF, metin parçalarına bölünerek ChromaDB’ye eklenir.
- Aynı dosyayı tekrar indekslemek isterseniz, çıkarıp yeniden ekleyebilir veya komutu yeniden çalıştırabilirsiniz (koleksiyon silinip baştan oluşturulur).
- PDF taranmış ise metin çıkarımı boş dönebilir; bu durumda OCR uygulamanız gerekir.

---

## Streamlit Uygulamasını Çalıştırma
```bash
streamlit run app.py
```

Arayüz bileşenleri:
- Sol menüde `Top K` (döndürülmek istenen bağlam parçası sayısı) ve çok dilli mod anahtarı.
- PDF listesi mevcut belgeleri gösterir.
- Form alanından soru gönderilir, iptal butonu uzun sorguları keser.
- Yanıtlar sonunda kullanılan kaynaklar `[1]`, `[2]` şeklinde listelenir.

---

## Streamlit Cloud’a Dağıtım
1. Depoyu GitHub’a (veya Streamlit Cloud’un erişebileceği bir kaynağa) gönderin.
2. Streamlit Cloud’da yeni bir uygulama oluştururken `app.py` dosyasını seçin.
3. **Secrets** bölümüne Gemini anahtarınızı ekleyin:
   ```toml
   GEMINI_API_KEY = "xxxxx"
   GEMINI_MODEL = "gemini-2.5-flash"
   ```
4. Uygulama açıldıktan sonra Cloud’ın Terminal sekmesinden bir defaya mahsus:
   ```bash
   python ingest.py --input data/
   ```
   komutunu çalıştırın. (Cloud ortamı kapatılıp yeniden açılırsa komutu tekrar çalıştırmanız gerekebilir.)
5. Veri gizliliği gerektiren PDF’ler için Streamlit Cloud’da `Secrets` veya `st.file_uploader` ile yükleme akışını tercih edin; bu repository’ye dosya koymaktan daha güvenlidir.

---

## Klasör Yapısı
```text
gaih-rag-chatbot/
├─ app.py              # Streamlit arayüzü ve entegrasyonlar
├─ rag_pipeline.py     # Hibrit retrieval + Gemini yanıt üretimi
├─ ingest.py           # PDF parçalama ve ChromaDB’ye ekleme
├─ assets/styles.css   # Streamlit teması
├─ data/               # PDF kaynakları (örnek dosyalarınızı buraya koyun)
├─ vectordb/           # Kalıcı Chroma koleksiyonu (ilk ingest sonrası oluşur)
├─ requirements.txt
└─ README.md
```

---

## Sık Karşılaşılan Sorunlar
- **`GEMINI_API_KEY` bulunamadı**: `.env` dosyasını oluşturup `streamlit run app.py` komutundan önce sanal ortamı aktifleştirdiğinizden emin olun.
- **`chromadb.errors.NotFoundError`**: Henüz indeks yok; `python ingest.py --input data/` komutunu çalıştırın.
- **Yanıtlar İngilizce geliyor**: Çok dilli modu kapatmayı deneyin. Sistem komutları Türkçe olsa da Gemini modelinin varsayılan davranışı modele göre değişebilir.
- **PDF metni boş görünüyor**: Dosya taranmış olabilir. OCR’den geçirdikten sonra tekrar ekleyin.
- **Streamlit Cloud’da dosyalar kayboluyor**: Her yeniden başlatmada `vectordb/` sıfırlanır; terminalden ingest komutunu tekrar çalıştırın veya başlatma betiğine ekleyin.

---

## Lisans
Kod tabanı MIT lisansı altındadır. Ayrıntılar için `LICENSE` dosyasına bakabilirsiniz.
