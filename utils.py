import os
import joblib

BASE = os.path.dirname(os.path.dirname(__file__))

model = joblib.load(os.path.join(BASE,"model","sentiment_ml_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE,"model","tfidf_vectorizer.pkl"))


class SentimentPredictor:

    def predict(self, text):
        x = vectorizer.transform([text])
        pred = model.predict(x)[0]
        prob = model.predict_proba(x).max()

        return {
            "sentiment": pred,
            "confidence": float(prob)
        }


def validate_input(text):

    if not isinstance(text,str) or not text.strip():
        return False, "Input must be non-empty text"

    if len(text) > 500:
        return False, "Text too long"

    return True, None