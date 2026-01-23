from matplotlib import pyplot as plt
import pandas as pd
import joblib
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix
from wordcloud import WordCloud
from feature_extraction import extract_features
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve




# # === Load dataset ===
df = pd.read_csv("phishing_site_urls.csv")
df.dropna(subset=["URL", "Label"], inplace=True)
print(df.shape)
df.head()
df.Label.value_counts()

#Plotting WordCloud
df_good=df[df.Label=='good']
df_bad=df[df.Label=='bad']

if not df_good.empty:
   good_url=" ".join(i for i in df_good.URL)
   wordcloud=WordCloud(width=1600,height=800,colormap='Paired').generate(good_url)
   plt.figure(figsize=(12,14),facecolor='k')
   plt.imshow(wordcloud,interpolation='bilinear')
   plt.axis('off')
   plt.title("Good URLs WordCloud")
   plt.tight_layout(pad=0)
   plt.show()
if not df_bad.empty:
   bad_url=" ".join(i for i in df_bad.URL)
   wordcloud=WordCloud(width=1600,height=800,colormap='Paired').generate(bad_url)
   plt.figure(figsize=(12,14),facecolor='k')
   plt.imshow(wordcloud,interpolation='bilinear')
   plt.axis('off')
   plt.title("Bad URLs WordCloud")
   plt.tight_layout(pad=0)
   plt.show()

# # === Encode labels ===
df['Label'] =  df['Label'].map({'bad': 1, 'good': 0})
print("Initial label counts:\n", df['Label'].value_counts())

# # === Feature Engineering ===
df = extract_features(df)
print("Available columns:", df.columns.tolist())

# Distribution 
sns.set(style="darkgrid")
ax=sns.countplot(y="Label",data=df,hue="use_of_ip")

sns.set(style="darkgrid")
ax=sns.countplot(y="Label",data=df,hue="abnormal_url")

sns.set(style="darkgrid")
ax=sns.countplot(y="Label",data=df,hue="google_index")

sns.set(style="darkgrid")
ax=sns.countplot(y="Label",data=df,hue="short_url")


# # === Balance dataset ===
min_size = min(df[df['Label'] == 1].shape[0], df[df['Label'] == 0].shape[0])
phish_df = df[df['Label'] == 1].sample(n=min_size, random_state=42)
legit_df = df[df['Label'] == 0].sample(n=min_size, random_state=42)
df = pd.concat([phish_df, legit_df]).sample(frac=1, random_state=42).reset_index(drop=True)
print("Balanced dataset distribution:\n", df['Label'].value_counts())

# === Define features ===
numeric_features = [
    'use_of_ip', 'abnormal_url', 'google_index', 'count_dot', 'count-www', 'count@',
    'count_dir', 'count_embed_domain', 'short_url', 'count-https', 'count-http',
    'count%', 'count?', 'count-', 'count=', 'url_length', 'hostname_length',
    'sus_url', 'fd_length', 'tld_length', 'count-digits', 'count-letters'
]

df[numeric_features].head()

df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors='coerce')
print("Missing values after conversion:\n", df[numeric_features].isnull().sum())
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
print("Label distribution after NaN drop:\n", df['Label'].value_counts())

# Then balance the dataset
min_size = min(df[df['Label'] == 1].shape[0], df[df['Label'] == 0].shape[0])
phish_df = df[df['Label'] == 1].sample(n=min_size, random_state=42)
legit_df = df[df['Label'] == 0].sample(n=min_size, random_state=42)
df = pd.concat([phish_df, legit_df]).sample(frac=1, random_state=42)
print("Balanced dataset label counts:\n", df['Label'].value_counts())

# === Vectorize URL text ===
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(df['URL'])
joblib.dump(vectorizer, 'vectorizer.pkl')

# === Normalize numeric features ===
scaler = MinMaxScaler()
X_numeric = scaler.fit_transform(df[numeric_features])
# pd.DataFrame(X_numeric, columns=numeric_features).head()
X_numeric = csr_matrix(X_numeric)
joblib.dump(scaler, 'url_numeric_scaler.pkl')

# === Combine features ===
X = hstack([X_vectorized, X_numeric])
y = df['Label']

print("Label distribution before split:\n", df['Label'].value_counts())

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Train Model ===
logistic_model = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced', random_state=42)
logistic_model.fit(X_train, y_train)
joblib.dump(logistic_model, 'phishing_logistic_model.pkl')


nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
joblib.dump(nb_model,"nb_model.pkl")



print("Model training complete. Files saved: phishing_logistic_model.pkl, nb_model.pkl, vectorizer.pkl")

print("Train label distribution:", y.value_counts())


# === Evaluation ===
print("\n--- Logistic Regression ---")
y_pred_logistic = logistic_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))

cm = confusion_matrix(y_test, y_pred_logistic)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# --- Logistic Regression: ROC and Precision-Recall ---
fpr, tpr, _ = roc_curve(y_test, y_pred_logistic)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_pred_logistic):.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid()
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_pred_logistic)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Logistic Regression")
plt.grid()
plt.show()


print("\n--- Naive Bayes ---")
y_pred_nb = nb_model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

cm = confusion_matrix(y_test, y_pred_nb)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Naive Bayes Confusion Matrix")
plt.show()

# --- Naive Bayes: ROC and Precision-Recall ---
fpr, tpr, _ = roc_curve(y_test, y_pred_nb)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_pred_nb):.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Naive Bayes")
plt.legend()
plt.grid()
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_pred_nb)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Naive Bayes")
plt.grid()
plt.show()


feature_names = vectorizer.get_feature_names_out()
coefs = logistic_model.coef_[0]
num_features = len(feature_names)

top_phish_idx = np.argsort(coefs)[-20:][::-1]
top_legit_idx = np.argsort(coefs)[:20]

print("\nTop indicative phishing tokens:")
for i in top_phish_idx:
    if i < num_features:
        print(f"{feature_names[i]}: {coefs[i]:.4f}")

print("\nTop indicative legitimate tokens:")
for i in top_legit_idx:
    if i < num_features:
        print(f"{feature_names[i]}: {coefs[i]:.4f}")



