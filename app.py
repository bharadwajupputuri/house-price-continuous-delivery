
from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
import math

app = Flask(__name__)

MODEL_PATH = "house_price_model.pkl"


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "House Price Prediction API is running",
        "endpoint": "/predict"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({
            "error": "Trained model not found"
        }), 500

    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not data:
        return jsonify({
            "error": "Send a valid JSON object containing house features"
        }), 400

    try:
        # Convert the JSON object into one row of input data.
        input_data = pd.DataFrame([data])

        # Load the trained pipeline and predict the house price.
        model = joblib.load(MODEL_PATH)
        prediction = float(model.predict(input_data)[0])

        if not math.isfinite(prediction):
            raise ValueError("Prediction is not a finite number")

        return jsonify({
            "predicted_sale_price": round(prediction, 2)
        }), 200

    except (ValueError, TypeError, KeyError) as error:
        return jsonify({
            "error": "Invalid house features",
            "details": str(error)
        }), 400

    except Exception:
        app.logger.exception("House price prediction failed")
        return jsonify({
            "error": "Prediction failed"
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
