# 📰 fakenws – Fake News Detection App

**fakenws** is a simple web application built with **Python** and **Streamlit** that detects whether a news article is **REAL** or **FAKE** using a machine learning model trained on labeled news data.

---

## 🚀 Features

- Detects fake vs real news from article text
- Built with Scikit-learn and Streamlit
- Fast prediction with confidence score
- Clean and easy-to-use interface

---

## 🧠 Model Overview

- **Algorithm**: PassiveAggressiveClassifier
- **Vectorizer**: TF-IDF (with stop words removed, max_df=0.7)
- **Dataset**: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Accuracy**: ~93% on test data

---

## ⚙️ How to Use

### 1. Clone the repository

```bash
git clone https://github.com/grahanurdian/fakenws.git
cd fakenws