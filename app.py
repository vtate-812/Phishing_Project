# from flask import Flask, render_template, request, jsonify
# import joblib
# import numpy as np
# import pandas as pd
# import re
# from googlesearch import search
# from scipy.sparse import hstack
# from urllib.parse import urlparse

# # Load the trained model and vectorizer
# def load_model():
#     return (
#         joblib.load('logistic_model.pkl'), 
#         joblib.load('nb_model.pkl'), 
#         joblib.load('vectorizer.pkl')  # ✅ Ensure this is the trained vectorizer
#     )

# logistic_model, nb_model, vectorizer = load_model()


# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
    
#     if 'URL' not in request.form:
#         return "Error: Missing 'URL' parameter in the request.", 400 
#     url = request.form['URL'].strip()

#     # ✅ Validate that URL is not empty
#     if not url:
#         return "Error: URL cannot be empty.", 400
    
#     # ✅ Log new URLs to a file for future retraining
#     # new_data = pd.DataFrame([[url, "unknown"]], columns=['URL', 'Label'])  # Mark as 'unknown'
#     # new_data.to_csv('new_urls.csv', mode='a', header=False, index=False)


#     # Extract features
#     features = vectorizer.transform([url])
#     hostname = urlparse(url).hostname if urlparse(url).hostname else ""
#     additional_features = np.array([[
#         1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,  #use_of_ip
#         1 if hostname in url else 0,  #abnormal_url
#         1 if search(url, 5) else 0,  # ✅ google_index (fix)
#         url.count('.'),  #count.
#         url.count('www'), #count-www
#         url.count('@'),  # count@
#         urlparse(url).path.count('/'),  # count_dir
#         urlparse(url).path.count('//'),  # count_embed_domian
#         1 if re.search(r'bit\.ly|goo\.gl|t\.co|tinyurl|tr\.im|is\.gd', url) else 0,  # short_url
#         url.count('https'),  #count-https
#         url.count('http'),   #count-http
#         url.count('%'),  # count%
#         url.count('?'),  # count?
#         url.count('-'),  # count-
#         url.count('='),  # count=
#         len(url),  #url_length
#         len(hostname),  #hostname_length
#         1 if re.search(r'paypal|login|signin|bank|account|update|free|bonus|ebayisapi', url) else 0,  # sus_url
#         len(urlparse(url).path.split('/')[1]) if len(urlparse(url).path.split('/')) > 1 else 0,  # fd_length
#         len(urlparse(url).netloc.split('.')[-1]) if '.' in urlparse(url).netloc else 0,  # tld_length
#         sum(c.isdigit() for c in url),  # count-digits
#         sum(c.isalpha() for c in url)  # count-letters
#     ]])


    
#     print(f"Vectorizer feature count: {vectorizer.get_feature_names_out().shape[0]}")
#     print(f"Features extracted from input: {features.shape[1]}")
#     print(f"Text Features: {features.shape[1]}")
#     print(f"Numerical Features: {additional_features.shape[1]}")
#     print(f"Total Features Expected: {logistic_model.coef_.shape[1]}")



#     additional_features = additional_features.reshape(1, -1)  # Ensures it remains 2D
#     # features_array = features.toarray()  # Convert sparse matrix to dense array
#     # Combine vectorized features and additional features
#     X = hstack([features, additional_features])

#     if X.shape[1] != logistic_model.coef_.shape[1]:
#         return f"Error: Feature mismatch! Expected {logistic_model.coef_.shape[1]}, got {X.shape[1]}."

#     # Predict
#     logistic_prediction = logistic_model.predict(X)[0]
#     nb_prediction = nb_model.predict(X)[0]
#     final_prediction = max(logistic_prediction, nb_prediction)
#     result = "Phishing Site Detected!" if final_prediction == 1 else "Legitimate Site."

#     return render_template('result.html', result=result)

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from urllib.parse import urlparse
import re
from scipy.sparse import hstack, csr_matrix
from tld import get_tld

from feature_extraction import extract_features

# Load models
logistic_model = joblib.load('phishing_logistic_model.pkl')
nb_model = joblib.load('nb_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
scaler = joblib.load('url_numeric_scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('URL', '').strip()
    if not url:
        return "Error: URL cannot be empty.", 400
    
    # try:
    #  google_index = 1 if any(re.search(url, num_results=5)) else 0
    # except:
    #  google_index = 0

    # # Feature extraction
    # features = vectorizer.transform([url])
    # hostname = urlparse(url).hostname if urlparse(url).hostname else ""
    # additional_features = np.array([[
    #     1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
    #     1 if hostname in url else 0,
    #     google_index,  # Static
    #     url.count('.'),
    #     url.count('www'),
    #     url.count('@'),
    #     urlparse(url).path.count('/'),
    #     urlparse(url).path.count('//'),
    #     1 if re.search(r'bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co', url) else 0,
    #     url.count('https'),
    #     url.count('http'),
    #     url.count('%'),
    #     url.count('?'),
    #     url.count('-'),
    #     url.count('='),
    #     len(url),
    #     len(hostname),
    #     1 if re.search(r'paypal|login|signin|bank|account|update|free|bonus|ebay', url) else 0,
    #     len(urlparse(url).path.split('/')[1]) if len(urlparse(url).path.split('/')) > 1 else 0,
    #     len(get_tld(url, fail_silently=True) or ''),
    #     sum(c.isdigit() for c in url),
    #     sum(c.isalpha() for c in url)

    # ]])
    


    # # Text + numeric features
    
    
    # print(f"Vectorizer feature count: {vectorizer.get_feature_names_out().shape[0]}")
    # print(f"Features extracted from input: {features.shape[1]}")
    # print(f"Text Features: {features.shape[1]}")
    # print(f"Numerical Features: {features.shape[1]}")
    # print(f"Total Features Expected: {logistic_model.coef_.shape[1]}")

    # additional_features = scaler.transform(additional_features)

    # # Combine vectorized features and additional features
    #  X = hstack([features, csr_matrix(additional_features)])

    df = pd.DataFrame({'URL': [url]})
    df = extract_features(df)
    numeric_features = [
        'use_of_ip', 'abnormal_url', 'google_index', 'count_dot', 'count-www', 'count@',
        'count_dir', 'count_embed_domain', 'short_url', 'count-https', 'count-http',
        'count%', 'count?', 'count-', 'count=', 'url_length', 'hostname_length',
        'sus_url', 'fd_length', 'tld_length', 'count-digits', 'count-letters'
    ]

    df[numeric_features] = df[numeric_features].fillna(0)
    X_text = vectorizer.transform(df['URL'])
    X_num = scaler.transform(df[numeric_features])
    X = hstack([X_text, csr_matrix(X_num)])



    if X.shape[1] != logistic_model.coef_.shape[1]:
        return f"Error: Feature mismatch! Expected {logistic_model.coef_.shape[1]}, got {X.shape[1]}."

    # Predict
    logistic_prediction = logistic_model.predict(X)[0]
    nb_prediction = nb_model.predict(X)[0]
    final_prediction = max(logistic_prediction, nb_prediction)
    result = "⚠️ Phishing Site Detected" if final_prediction == 1 else "✅ Legitimate Site"

    return render_template('result.html', result=result)

    

if __name__ == '__main__':
    app.run(debug=True)
