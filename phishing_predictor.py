import pandas as pd
import numpy as np
import joblib
import re
from urllib.parse import urlparse
from sklearn.metrics import confusion_matrix
from tld import get_tld
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, csr_matrix

# ---------------------- Load Assets ----------------------
vectorizer = joblib.load('vectorizer.pkl')
scaler = joblib.load('url_numeric_scaler.pkl')
logistic_model = joblib.load('phishing_logistic_model.pkl')
nb_model = joblib.load('nb_model.pkl')


# ---------------------- Feature Extraction ----------------------
def extract_features(df):
    def having_ip(url):
        return 1 if re.search(r"\d+\.\d+\.\d+\.\d+", str(url)) else 0

    def abnormal_url(url):
        hostname = urlparse(url).hostname or ''
        return 1 if hostname in url else 0

    def count_dir(url): return urlparse(url).path.count('/')
    def count_embed(url): return urlparse(url).path.count('//')
    def shortening_service(url):
        return 1 if re.search(r'bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co', str(url)) else 0
    def suspicious_words(url):
        return 1 if re.search(r'login|signin|bank|account|update|paypal|free|bonus|ebay', str(url)) else 0

    df['use_of_ip'] = df['URL'].apply(having_ip)
    df['abnormal_url'] = df['URL'].apply(abnormal_url)
    df['google_index'] = 0
    df['count_dot'] = df['URL'].apply(lambda x: str(x).count('.'))
    df['count-www'] = df['URL'].apply(lambda x: str(x).count('www'))
    df['count@'] = df['URL'].apply(lambda x: str(x).count('@'))
    df['count_dir'] = df['URL'].apply(count_dir)
    df['count_embed_domain'] = df['URL'].apply(count_embed)
    df['short_url'] = df['URL'].apply(shortening_service)
    df['count-https'] = df['URL'].apply(lambda x: str(x).count('https'))
    df['count-http'] = df['URL'].apply(lambda x: str(x).count('http'))
    df['count%'] = df['URL'].apply(lambda x: str(x).count('%'))
    df['count?'] = df['URL'].apply(lambda x: str(x).count('?'))
    df['count-'] = df['URL'].apply(lambda x: str(x).count('-'))
    df['count='] = df['URL'].apply(lambda x: str(x).count('='))
    df['url_length'] = df['URL'].apply(lambda x: len(str(x)))
    df['hostname_length'] = df['URL'].apply(lambda x: len(urlparse(x).netloc))
    df['sus_url'] = df['URL'].apply(suspicious_words)
    df['fd_length'] = df['URL'].apply(lambda x: len(urlparse(x).path.split('/')[1]) if len(urlparse(x).path.split('/')) > 1 else 0)
    df['tld_length'] = df['URL'].apply(lambda x: len(get_tld(x, fail_silently=True) or ''))
    df['count-digits'] = df['URL'].apply(lambda x: sum(c.isdigit() for c in str(x)))
    df['count-letters'] = df['URL'].apply(lambda x: sum(c.isalpha() for c in str(x)))
    return df

# ---------------------- Prediction ----------------------
def predict_urls(urls, model_choice="logistic", threshold=0.5):
    df = pd.DataFrame({'URL': urls})
    df = extract_features(df)

    # Debugging: Check available columns
    print("Available columns after extraction:", df.columns.tolist())

    numeric_features = [
        'use_of_ip', 'abnormal_url', 'google_index', 'count_dot', 'count-www', 'count@',
        'count_dir', 'count_embed_domain', 'short_url', 'count-https', 'count-http',
        'count%', 'count?', 'count-', 'count=', 'url_length', 'hostname_length',
        'sus_url', 'fd_length', 'tld_length' ,'count-digits', 'count-letters'
    ]

    # Ensure all numeric features are present in df
    missing_features = [feature for feature in numeric_features if feature not in df.columns]
    if missing_features:
        print(f"Warning: Missing features in DataFrame: {missing_features}")

    df[numeric_features] = df[numeric_features].fillna(0)   
    X_text = vectorizer.transform(df['URL'])
    X_numeric = scaler.transform(df[numeric_features].fillna(0))
    X_combined = hstack([X_text, csr_matrix(X_numeric)])

    if model_choice == "naive_bayes":
        y_probs = nb_model.predict_proba(X_combined)[:, 1]
        preds = (y_probs >= 0.5).astype(int)

    else:
        # Get predicted probabilities for Logistic Regression
        y_probs = logistic_model.predict_proba(X_combined)[:, 1]  # Probabilities for class 1 (phishing)
        preds = (y_probs >= 0.5).astype(int)  # Apply thresholding for classification




    # logistic_preds = logistic_model.predict(X_combined)
    return list(zip(urls, preds))





# ---------------------- Example ----------------------
if __name__ == '__main__':
    urls = [
        'http://secure-login-paypal.com/verify',
    'http://bit.ly/ebay-login',
    'http://freebonusbankupdate.xyz/account/verify',
    'http://google.com-security-check.ml/login',
    'http://login-update.facebook.com-sessionverify.ru',
    'https://github.com',  # legit
    'https://www.amazon.com'  # legit
    ]

    print("Logistic Regression Predictions:")
    for url, label in predict_urls(urls, model_choice='logistic', threshold=0.5):
        print(f"{url} => {'Phishing' if label == 1 else 'Legit'}")
    
    print("\nNaive Bayes Predictions:")
    for url, label in predict_urls(urls, model_choice='naive_bayes'):
        print(f"{url} => {'Phishing' if label == 1 else 'Legit'}")

    for url, label in predict_urls(urls):
        print(f"{url} => {'Phishing' if label == 1 else 'Legit'}")
