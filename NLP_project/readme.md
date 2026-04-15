# 🌐 Multi-Modal AI Translator Pro

A Final Year Major Project built using **Transformer-based NLP, OCR, Speech Recognition, and Text-to-Speech** to provide real-time multilingual translation across text, voice, image, and documents.

---

## 🚀 Features

* ✏ **Text Translation**
* 🖼 **Image OCR Translation**
* 🎤 **Voice Translation**
* 📄 **Document Translation**
* 💬 **Conversation Translator**
* 📜 Translation History Tracking
* ⭐ Favorite Translations
* 📊 Performance Metrics Dashboard

---

## 🤖 AI / ML Models Used

### Translation Model

* **facebook/nllb-200-distilled-600M**
* Transformer-based Multilingual Sequence-to-Sequence Model
* Developed by Meta AI
* Supports 200+ Languages
* Used for high-quality multilingual neural machine translation

### OCR Model

* **EasyOCR**
* Used for extracting text from uploaded images / camera input

### Speech Recognition

* **Google Speech Recognition API**
* Converts spoken audio into text

### Text-to-Speech

* **gTTS (Google Text-to-Speech)**
* Converts translated text into speech output

---

## 🛠 Tech Stack

* **Frontend/UI:** Streamlit
* **Backend:** Python
* **Database:** SQLite
* **Deep Learning Framework:** PyTorch
* **Transformer Library:** Hugging Face Transformers

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/Multi-Modal-AI-Translator.git
cd Multi-Modal-AI-Translator
pip install -r requirements.txt
streamlit run app.py
```

---

## 📋 Requirements

See `requirements.txt`

---

## 🌍 Supported Languages

English, Hindi, German, French, Spanish, Chinese, Japanese, Tamil, Telugu, Malayalam, Kannada, Korean

---

## ☁ Deployment

Deployed using **Streamlit Community Cloud**

---

## 📈 Performance Highlights

* Supports Multi-Modal Input:

  * Text
  * Audio
  * Images
  * Documents

* Includes:

  * Translation Accuracy Dashboard
  * Language-wise Performance Evaluation
  * Real-Time Latency Measurement

---

## 📂 Project Structure

```bash
Multi-Modal-AI-Translator/
│
├── app.py
├── requirements.txt
├── packages.txt
├── runtime.txt
├── translator_data.db
└── README.md
```

---

## 👨‍💻 Author

Final Year Major Project
Developed for Academic Submission

---

## 📜 License

This project is for educational and academic purposes.
