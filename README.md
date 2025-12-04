🌟 Twitter Sentiment Analyzer — Explainable AI Dashboard

Advanced NLP • TF-IDF Logistic Regression • Streamlit • Explainability

<p align="center"> <img src="https://img.shields.io/badge/NLP-Sentiment%20Analysis-blueviolet?style=for-the-badge"/> <img src="https://img.shields.io/badge/Model-Logistic%20Regression-brightgreen?style=for-the-badge"/> <img src="https://img.shields.io/badge/Explainable%20AI-Token%20Attribution-orange?style=for-the-badge"/> <img src="https://img.shields.io/badge/UI-Streamlit-red?style=for-the-badge"/> </p>

🚀 Project Overview



A fully interactive Sentiment Analysis Dashboard built using TF-IDF + Logistic Regression, enhanced with Explainable AI and a modern Streamlit UI.



This app allows users to:



✔ Analyze sentiment of any text

✔ Upload CSV files for batch processing

✔ View visual insights (bar charts, word clouds)

✔ See which words contributed most to positive/negative sentiment

✔ Get instant, shareable insights on a clean and modern dashboard



🔗 Live Demo: <your Streamlit URL here>

📦 Repository: https://github.com/M1325-source/Sentiment-Analysis



✨ Key Features

🔍 1. Real-Time Text Sentiment Analysis



Enter any text or paragraph



Model predicts:



Sentiment → Positive / Negative



Confidence score



Beautiful “confidence gauge bar”



📊 2. CSV Report Studio (Batch Analysis)



Upload a CSV with a text column and get:



Full predictions table



Sentiment distribution chart



Positive + Negative word clouds



Dataset summary metrics



Perfect for analyzing tweets, reviews, support messages, etc.



🧠 3. Explainable AI (Token-Level Influence)



For every input text, the app highlights:



Top positive words boosting sentiment



Top negative words lowering sentiment



Weighted importance score for each word



This makes the model transparent \& recruiter-impressive.



🎨 4. Modern, Dark-Themed UI



Professional dashboard feel



Clean typography



Animated components



Two-tab layout:



Live Text Analyzer



CSV Report Studio



🏗️ Tech Stack

Layer	Technology

Frontend UI	Streamlit (Custom CSS + Responsive Layout)

Machine Learning Model	Logistic Regression

Feature Engineering	TF-IDF Vectorizer

Explainability	Token Weight Attribution

Visualizations	Matplotlib, WordCloud

Language Processing	NLTK

Deployment	Streamlit Cloud

📁 Project Structure

Sentiment-Analysis/

│

├── app.py                     # Streamlit Dashboard

├── sentiment\_utils.py         # Prediction + Explainability Engine

├── train\_model.py             # Model training script

├── requirements.txt           # Dependencies

│

├── models/

│   ├── sentiment\_model.joblib

│   └── tfidf\_vectorizer.joblib

│

└── test\_analysis.py           # Local testing script



⚙️ How to Run Locally

1️⃣ Clone repository

git clone https://github.com/M1325-source/Sentiment-Analysis.git

cd Sentiment-Analysis



2️⃣ Install dependencies

pip install -r requirements.txt



3️⃣ Run the app

streamlit run app.py



🎯 Model Workflow

Raw Text → Cleaning → TF-IDF Vectorization → Logistic Regression Model

&nbsp;                                     ↓

&nbsp;                            Explainable Token Weights

&nbsp;                                     ↓

&nbsp;                          Interactive Streamlit Dashboard

