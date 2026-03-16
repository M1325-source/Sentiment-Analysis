# 🌟 Twitter Sentiment Analyzer — Explainable AI Dashboard

> Advanced NLP pipeline using **TF-IDF + Logistic Regression** with Explainable AI token attribution, batch CSV processing, and a live Streamlit dashboard.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B)](https://sentiment-analysis-58bkiqua6elowgtt5wnkjc.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-100%25-blue)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/M1325-source/Sentiment-Analysis)](https://github.com/M1325-source/Sentiment-Analysis)

---

## 🔗 Live Demo

👉 **[Try it live →](https://sentiment-analysis-58bkiqua6elowgtt5wnkjc.streamlit.app/)**

Type any text or upload a CSV and get instant sentiment predictions with explainability!

---

## 🎯 Overview

A fully interactive **Sentiment Analysis Dashboard** that goes beyond basic positive/negative prediction — it shows **which words drove the sentiment** using Explainable AI token attribution.

Built for:
- Social media analysis (tweets, reviews)
- Customer feedback classification
- Batch processing of large text datasets

---

## ✨ Features

### 🔍 Real-Time Text Analysis
- Enter any text and get instant prediction
- **Positive / Negative** classification
- **Confidence score** with visual gauge bar

### 📊 CSV Batch Processing (Report Studio)
- Upload any CSV with a text column
- Full predictions table with sentiment labels
- Sentiment distribution bar chart
- Positive + Negative **word clouds**
- Dataset summary metrics

### 🧠 Explainable AI (Token Attribution)
- Highlights **top positive words** boosting sentiment
- Highlights **top negative words** lowering sentiment
- Weighted importance score per word
- Makes the model **transparent and interpretable**

### 🎨 Modern Dark-Themed UI
- Professional dashboard layout
- Two-tab interface: Live Analyzer + CSV Studio
- Clean typography and animated components

---

## 🏗️ System Architecture
```
Raw Text Input
      │
      ▼
Text Cleaning (NLTK)
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Logistic Regression Model
      │
      ├──► Sentiment Prediction (Positive/Negative)
      ├──► Confidence Score
      └──► Token Weight Attribution (Explainability)
                  │
                  ▼
         Streamlit Dashboard
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| ML Model | Logistic Regression (scikit-learn) |
| Feature Engineering | TF-IDF Vectorizer |
| Explainability | Token Weight Attribution |
| NLP Preprocessing | NLTK |
| Visualizations | Matplotlib, WordCloud |
| UI & Deployment | Streamlit + Streamlit Cloud |

---

## 📁 Project Structure
```
Sentiment-Analysis/
│
├── app.py                     ← Streamlit dashboard (main entry)
├── sentiment_utils.py         ← Prediction + Explainability engine
├── train_model.py             ← Model training script
├── test_analysis.py           ← Local testing script
├── requirements.txt
│
└── models/
    ├── sentiment_model.joblib     ← Trained model
    └── tfidf_vectorizer.joblib    ← Fitted vectorizer
```

---

## 🚀 Run Locally
```bash
# Clone the repo
git clone https://github.com/M1325-source/Sentiment-Analysis.git
cd Sentiment-Analysis

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Algorithm | Logistic Regression |
| Features | TF-IDF (n-grams) |
| Dataset | Twitter Sentiment140 |
| Accuracy | ~85%+ |

---

## 🔮 Future Enhancements

- BERT / Transformer-based model upgrade
- Multi-class sentiment (Very Positive, Neutral, Very Negative)
- Real-time Twitter API integration
- Aspect-based sentiment analysis
- REST API endpoint for programmatic access

---

## 👩‍💻 Author

**Manisha Priya** — AI/ML Engineer
- 🐙 GitHub: [@M1325-source](https://github.com/M1325-source)
- 📧 Email: manishapriya1325@gmail.com

⭐ **Star this repo if you found it useful!**
