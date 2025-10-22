# Kali Linux RAG Chatbot

Gemini destekli **Retrieval-Augmented Generation** (RAG) chatbotu. Kali Linux dokümantasyonunu içeren PDF’leri ChromaDB üzerinde indeksleyip Streamlit arayüzüyle Türkçe yanıt ve kaynak referansı üretir. İsteğe bağlı çok dilli mod sayesinde Türkçe dışındaki sorular, daha doğru geri getirme için Gemini ile İngilizceye yeniden yazılır.

---

## Özellikler
- **Gemini-only**: Yanıtlar Google Gemini API üzerinden üretilir; ek LLM entegrasyonu gerekmez.
- **Hibrit retrieval**: ChromaDB üzerinde hem vektör benzerliği hem de anahtar kelime eşleşmesi kullanılır.
- **Önceden hazırlanmış vektör DB**: `vectordb/` klasörüyle birlikte dağıtabilir, sunucuda ingest çalıştırmadan hazır indeksle açabilirsiniz.
- **Streamlit arayüzü**: Top-K seçimi, çok dilli mod anahtarı, kaynak listesi, sohbet geçmişi ve iptal özelliği.
- **Sohbet dışa aktarma**: Oturumu `.txt` olarak indirebilirsiniz.

---

## Başlangıç
### Gereksinimler
- Python 3.10+
- Google Gemini API anahtarı (Google AI Studio)
- `pip`, `virtualenv` (opsiyonel ama önerilir)

### Kurulum
1. **Depoyu alın**
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
4. **.env dosyasını ekleyin**
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   GEMINI_MODEL=gemini-2.5-flash   # opsiyonel, varsayılan değer
   ```

> ⚠️ `GEMINI_API_KEY` olmadan uygulama başlatılamaz.

---

## Vektör Veritabanını Hazırlama
Uygulama açılırken `vectordb/` içinde `docs` koleksiyonunu arar. İki farklı yaklaşım kullanabilirsiniz:

1. **Önceden oluşturulmuş indeksle dağıtım (önerilen)**  
   - Yerelde PDF’leri indeksleyip `vectordb/` klasörünü repoda tutun (SQLite dosyası artık `.gitignore` dışına alındı).  
   - Sunucuya kodu gönderdiğinizde uygulama hazır koleksiyonu kullanır; ingest komutuna gerek kalmaz.

2. **Sunucuda ingest çalıştırma**  
   - `data/` klasörüne PDF’leri koyun.  
   - Aşağıdaki komutu çalıştırarak yeni koleksiyon oluşturun:
     ```bash
     python ingest.py --input data/
     ```
   - Komut, mevcut koleksiyonu silip baştan oluşturur. PDF taranmış ise metin çıkarımı boş dönebilir; bu durumda önce OCR uygulayın.

> Koleksiyon dosyaları büyük olabilir. Streamlit Cloud gibi ortamlarda kota sınırlarını kontrol edin.

---

## Streamlit Uygulamasını Çalıştırma
```bash
streamlit run app.py
```

Arayüzde:
- Sol sidebar’da `Top K` (bağlam parçası sayısı) ve çok dilli mod anahtarı bulunur.
- Chroma koleksiyonunun toplam parça sayısı ve veri kaynakları listelenir.
- Form alanından soru gönderilir; “İptal Et” butonu uzun sorguları sonlandırır.
- Yanıtlar kullanılan kaynakları `[1]`, `[2]` formatıyla gösterir.
- Sohbet geçmişi sayfanın altına doğru listelenir ve `.txt` olarak indirilebilir.

---

## Dağıtım Notları (Streamlit Cloud Örneği)
1. Depoyu `vectordb/` klasörüyle birlikte GitHub’a gönderin.
2. Streamlit Cloud’da uygulamayı oluşturun ve `app.py` dosyasını seçin.
3. **Secrets** bölümüne Gemini anahtarınızı ekleyin:
   ```toml
   GEMINI_API_KEY = "xxxxx"
   GEMINI_MODEL = "gemini-2.5-flash"
   ```
4. Uygulama açıldığında hazır koleksiyon yüklenir. Koleksiyon göndermediyseniz terminalden
   ```bash
   python ingest.py --input data/
   ```
   komutunu çalıştırın.
5. Ortam yeniden başlatıldığında `vectordb/` klasörü korunmuyorsa (ör. ephemeral disk), ingest komutunu otomatik başlatmak için `startup.sh` benzeri bir betik kullanın veya koleksiyonu her dağıtımda yeniden yükleyin.

---

## Klasör Yapısı
```text
gaih-rag-chatbot/
├─ app.py              # Streamlit arayüzü
├─ rag_pipeline.py     # Hibrit retrieval + Gemini çağrısı
├─ ingest.py           # PDF parçalama ve Chroma ingest
├─ assets/styles.css   # Streamlit teması
├─ data/               # Kaynak PDF’ler
├─ vectordb/           # Kalıcı Chroma koleksiyonu
├─ requirements.txt
└─ README.md
```

---

## Sık Sorulanlar
- **“Chroma koleksiyonu bulunamadı” uyarısı alıyorum.**  
  `vectordb/` klasörünü dağıtıma dahil ettiğinizden emin olun veya `python ingest.py --input data/` ile koleksiyon oluşturun.
- **Yanıtlar İngilizce geliyor.**  
  Sidebar’daki çok dilli modu kapatın. Gemini modeli bazen bağlamı İngilizce yanıtlayabilir.
- **PDF metni boş geliyor.**  
  Dosya taranmış olabilir; OCR uygulayıp tekrar ingest edin.
- **İlk sorgu yavaş.**  
  SentenceTransformer modeli ilk kullanımda indiriliyor. Dağıtımdan sonra kısa bir “ısınma” sorgusu çalıştırmak açılış süresini iyileştirir.

---

## Lisans
Bu proje MIT lisansı ile dağıtılır. Ayrıntılar için `LICENSE` dosyasına bakabilirsiniz.

## Deploy Link
https://kaliragchat.streamlit.app/