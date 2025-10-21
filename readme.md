# 🤖 TR-FAQ RAG Chatbot  

Akbank Generative AI Bootcamp için hazırlanmış Türkçe **RAG (Retrieval-Augmented Generation)** tabanlı chatbot projesi.  
PDF belgelerinden bilgi çekip kaynaklı yanıtlar üreten, çok dilli destekli bir AI asistanıdır.  

---

## 📋 Proje Hakkında  

Bu proje, `data/` klasöründeki PDF’leri okuyarak vektör veritabanına işler ve  
**Gemini** veya **OpenAI GPT** modellerinden biriyle kullanıcı sorularına akıllı, kaynaklı yanıtlar döndürür.  

İsteğe bağlı “Çok Dilli Mod” aktif edildiğinde sistem yabancı dillerde gelen soruları otomatik olarak İngilizce’ye çevirir,  
yanıtı Türkçe üretir ve kullanıcıya sunar.  

---

## 🛠️ Kullanılan Teknolojiler  

- **Streamlit** – Web arayüzü  
- **LangChain + Sentence-Transformers** – Metin embedding işlemleri  
- **ChromaDB** – Vektör veritabanı  
- **Google Gemini / OpenAI GPT** – LLM yanıt üretimi  
- **python-dotenv** – Ortam değişkenleri yönetimi  
- **pypdf** – PDF metin çıkarma  

---

## 🚀 Kurulum  

### 1️⃣ Gerekli Paketleri Yükleyin  
```bash
1️⃣ Sanal ortam oluşturun
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

2️⃣ Gereken paketleri kurun
pip install -r requirements.txt

3️⃣ API Anahtarlarını Ayarlayın

Proje kök dizininde .env dosyası oluşturun:

GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

🔑 Gemini API key: Google AI Studio
🔑 OpenAI API key: platform.openai.com

4️⃣ Uygulamayı Çalıştırın
streamlit run app.py

📁 Proje Yapısı
GAIH-RAG-CHATBOT/
│
├── app.py
├── ingest.py
├── rag_pipeline.py
│
├── assets/
│   └── styles.css
│
├── data/
│   ├── eBay-Block-category-list.pdf
│   └── ...
│
├── vectordb/
├── .streamlit/
│   └── config.toml
│
├── .env.example
├── requirements.txt
├── to-do.md
├── LICENSE
└── README.md
```
### 💡 Nasıl Çalışır?

PDF Yükleme: data/ klasörüne PDF eklenir veya arayüzden yüklenir.

İndeksleme: PDF’ler parçalara bölünür, embedding ile vektör haline getirilir.

Sorgulama: Kullanıcı sorusu embedding’e dönüştürülür, en ilgili parçalar bulunur.

Yanıt Üretimi: Seçilen model (Gemini / GPT) kaynak metinlere göre yanıt üretir.

Çıktı: Yanıt + kaynak PDF dosya adları ve sayfa numaraları gösterilir.

🎯 Örnek Sorular

“Category ID 171146 olan nedir?”

“Learning-Kali-Linux PDF’inde packet sniffing hangi sayfada?”

“Organizational Behavior PDF’inde liderlik tanımı ne?”

⚙️ Çok Dilli Mod

🗣️ “Sorguyu İngilizceye yeniden yaz” seçeneği açıkken:
Kullanıcının yazdığı soru İngilizce’ye çevrilir → Model çalışır → Yanıt Türkçe döndürülür.

Bu özellik yabancı dilde yazılmış soruların daha iyi anlaşılmasını sağlar,
ancak modelin cevabı birkaç saniye gecikebilir.

⚠️ Önemli Notlar

İlk çalıştırmada modeller indirilir, 1 defaya mahsustur.

Büyük PDF'lerde uzun süren indeksleme gözlemlenebilir.

Aynı dosya ismine sahip yüklemeler otomatik olarak tekrar yazılmaz.

Çok dilli mod ek işlem süresi gerektirebilir.

### 🔧 Modüler Yapı

Dosya	Açıklama
app.py	Streamlit tabanlı arayüz
rag_pipeline.py	Sorgu işleme ve model çağrısı
ingest.py	PDF metinlerini parçalayıp ChromaDB’ye kaydeder
assets/styles.css	UI renk, tema, ikon düzenlemeleri

### 🧩 Sorun Giderme
Hata	Çözüm
ModuleNotFoundError	pip install -r requirements.txt çalıştırın
API key expired / invalid	.env dosyasındaki anahtarları yenileyin
chromadb.errors.NotFoundError	python ingest.py --input data/ çalıştırın
Gemini model not found	.env dosyasına GEMINI_MODEL=gemini-2.5-flash ekleyin
FileNotFoundError	data/ klasöründe PDF olduğundan emin olun
### 🧠 Geliştirme Notları

 PDF yükleme & otomatik indeksleme

 Gemini ve GPT desteği

 Türkçe ve çok dilli destek

 Dinamik PDF listesi

 Responsive arayüz (mobil görünüm)

 Chat geçmişi kaydı

 SQLite tabanlı sorgu arşivi

### 📝 Lisans

Bu proje eğitim amaçlıdır.
Kodlar MIT lisansı altındadır.
