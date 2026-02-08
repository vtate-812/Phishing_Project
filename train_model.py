import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix
from feature_extraction import extract_features

# === Load dataset ===
df = pd.read_csv("phishing_site_urls.csv")
df.dropna(subset=["URL", "Label"], inplace=True)

df['Label'] = df['Label'].map({'bad': 1, 'good': 0})
print("Initial label distribution:\n", df['Label'].value_counts())

# === Feature extraction ===
df = extract_features(df)

numeric_features = [
    'use_of_ip','abnormal_url','google_index','count_dot','count-www','count@',
    'count_dir','count_embed_domain','short_url','count-https','count-http',
    'count%','count?','count-','count=','url_length','hostname_length',
    'sus_url','fd_length','tld_length','count-digits','count-letters'
]

df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors='coerce')
df[numeric_features] = df[numeric_features].fillna(0)

# === Balance dataset (ONCE) ===
min_size = min(df[df.Label == 1].shape[0], df[df.Label == 0].shape[0])
df = pd.concat([
    df[df.Label == 1].sample(min_size, random_state=42),
    df[df.Label == 0].sample(min_size, random_state=42)
]).sample(frac=1, random_state=42)

# === URL Vectorizer (CHAR N-GRAM BEST PRACTICE) ===
vectorizer = CountVectorizer(
    analyzer="char",
    ngram_range=(3,5),
    max_features=5000
)
X_text = vectorizer.fit_transform(df['URL'])
joblib.dump(vectorizer, "vectorizer.pkl")

# === Scale numeric features ===
scaler = MinMaxScaler()
X_num = scaler.fit_transform(df[numeric_features])
X_num = csr_matrix(X_num)
joblib.dump(scaler, "url_numeric_scaler.pkl")

# === Combine features ===
X = hstack([X_text, X_num])
y = df['Label']

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Train Logistic Regression ===
model = LogisticRegression(
    solver="liblinear",
    max_iter=300,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
joblib.dump(model, "phishing_logistic_model.pkl")

# === Evaluation ===
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("✅ Training complete. Models saved.")
