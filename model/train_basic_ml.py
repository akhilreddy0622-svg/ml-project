import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE, "dataset", "sentiment.csv")

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["text","label"])
df["text"] = df["text"].astype(str).str.lower().str.strip()
df["label"] = df["label"].astype(str).str.lower().str.strip()
df = df.drop_duplicates(subset="text")

valid_labels = ["positive","negative","neutral"]
df = df[df["label"].isin(valid_labels)]

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=40000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9,
    stop_words="english",
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

MODEL_DIR = os.path.join(BASE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "sentiment_ml_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))