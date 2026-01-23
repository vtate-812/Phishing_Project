# 🛡️ SafeNet – Proactive Phishing Page Detection

A machine learning–based web application that detects phishing websites by analyzing URL characteristics. The system classifies URLs as **legitimate** or **phishing** using trained ML models and provides results through a user-friendly web interface.

---

## 🚀 Features

- Detects phishing websites using URL-based feature analysis
- Supports **Logistic Regression** and **Multinomial Naive Bayes**
- Displays prediction probability for better interpretability
- Simple and interactive **Flask** web interface
- Efficient model loading using **joblib**

---

## 🧠 Machine Learning Approach

### 🔹 Feature Engineering
- Extracts URL features such as:
  - IP address usage
  - URL length
  - Suspicious keywords
  - Special characters
  - Directory depth
  - Top-Level Domain (TLD) length

### 🔹 Text Processing
- URL tokenization using **CountVectorizer**

### 🔹 Models Used
- Logistic Regression
- Multinomial Naive Bayes

---

## 🛠️ Technology Stack

- **Python** – Core programming language
- **Pandas & NumPy** – Data preprocessing and feature extraction
- **scikit-learn** – Machine learning models
- **joblib** – Model serialization and loading
- **Flask** – Web framework
- **HTML & CSS** – Frontend development
- **Matplotlib** – Data visualization

---

## 📁 Project Structure
```
Phishing_Project/
│
├── app.py                     # Flask application
├── feature_extraction.py      # URL feature extraction logic
├── vectorizer.pkl             # Trained CountVectorizer
├── url_numeric_scaler.pkl     # Scaler for numeric features
├── phishing_logistic_model.pkl
├── nb_model.pkl
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── style.css
├── requirements.txt
└── README.md
```