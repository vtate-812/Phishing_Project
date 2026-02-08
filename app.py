from flask import Flask, render_template, request
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from feature_extraction import extract_features

app = Flask(__name__)

# === Load models ===
model = joblib.load("phishing_logistic_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("url_numeric_scaler.pkl")

numeric_features = [
    'use_of_ip','abnormal_url','google_index','count_dot','count-www','count@',
    'count_dir','count_embed_domain','short_url','count-https','count-http',
    'count%','count?','count-','count=','url_length','hostname_length',
    'sus_url','fd_length','tld_length','count-digits','count-letters'
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    url = request.form.get("URL", "").strip()
    if not url:
        return "URL cannot be empty", 400

    df = pd.DataFrame({"URL": [url]})
    df = extract_features(df)
    df[numeric_features] = df[numeric_features].fillna(0)

    X_text = vectorizer.transform(df["URL"])
    X_num = scaler.transform(df[numeric_features])
    X = hstack([X_text, csr_matrix(X_num)])

    proba = model.predict_proba(X)[0][1]

    if proba >= 0.6:
        result = f"⚠️ Phishing Website ({proba:.2f})"
    else:
        result = f"✅ Legitimate Website ({proba:.2f})"

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
