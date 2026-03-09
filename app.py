from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import SentimentPredictor, validate_input

app = Flask(__name__)
CORS(app)

predictor = SentimentPredictor()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    text = data.get("text","")

    ok, err = validate_input(text)
    if not ok:
        return jsonify({"success":False,"message":err}), 400

    result = predictor.predict(text)

    return jsonify({
        "success": True,
        "sentiment": result["sentiment"],
        "confidence": result["confidence"]
    })


@app.route("/health")
def health():
    return jsonify({"success":True,"status":"ok"})


if __name__ == "__main__":
    app.run(debug=True)