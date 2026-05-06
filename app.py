# =========================================================
# IMPORTS
# =========================================================
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# LOAD DATA (ROBUST)
# =========================================================
def load_data(path):
    for enc in ["utf-8","latin-1","cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except:
            continue
    raise ValueError("Error loading file")

data = load_data("twitter.csv")


# =========================================================
# AUTO COLUMN DETECTION (SIMPLE)
# =========================================================
def detect_columns(df):
    text_col, label_col = None, None

    for col in df.columns:
        c = col.lower()

        if any(k in c for k in ["text","tweet","review","content"]):
            text_col = col

        if any(k in c for k in ["sentiment","label","target"]):
            label_col = col

    if text_col is None:
        text_col = df.select_dtypes(include="object").columns[0]

    print("Text column:", text_col)
    print("Label column:", label_col)

    return text_col, label_col

text_col, label_col = detect_columns(data)


# =========================================================
# CLEANING (FAST - NO NLTK)
# =========================================================
def clean_text(series):
    return (
        series.astype(str)
        .str.lower()
        .str.replace(r"http\S+","",regex=True)
        .str.replace(r"[@#]\w+","",regex=True)
        .str.replace(r"[^a-z\s]"," ",regex=True)
        .str.replace(r"\s+"," ",regex=True)
        .str.strip()
    )


# =========================================================
# PREPROCESS (FAST)
# =========================================================
data = data.dropna(subset=[text_col])
data = data.drop_duplicates()

data["clean_text"] = clean_text(data[text_col])
data = data[data["clean_text"] != ""]


# =========================================================
# LABEL ENCODING
# =========================================================
le = LabelEncoder()
data["sentiment_enc"] = le.fit_transform(data[label_col])

print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))


# =========================================================
# TF-IDF (OPTIMIZED)
# =========================================================
vectorizer = TfidfVectorizer(
    max_features=15000,     # reduced for speed
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X = vectorizer.fit_transform(data["clean_text"])
Y = data["sentiment_enc"]

print("Feature shape:", X.shape)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)


# =========================================================
# MODEL (FASTER)
# =========================================================
model = LinearSVC(C=1.0, max_iter=1500)
model.fit(X_train, Y_train)


# =========================================================
# EVALUATION
# =========================================================
Y_pred = model.predict(X_test)

acc = accuracy_score(Y_test, Y_pred)
print(f"\nAccuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred, target_names=le.classes_))


# =========================================================
# CONFUSION MATRIX
# =========================================================
cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_sentiment(text):
    cleaned = clean_text(pd.Series([text]))
    vec = vectorizer.transform(cleaned)
    pred = model.predict(vec)[0]
    return le.inverse_transform([pred])[0]


# =========================================================
# TEST
# =========================================================
samples = [
    "I absolutely love this game!",
    "This is terrible and broken",
    "It's okay, nothing special",
    "Borderlands is fun but repetitive"
]

for s in samples:
    print(f"{s} → {predict_sentiment(s)}")
