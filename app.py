from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/flood_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]

    scaled_data = scaler.transform([features])

    prediction = model.predict(scaled_data)[0]

    return render_template(
        "result.html",
        prediction=float(prediction)
    )

if __name__ == "__main__":
    app.run(debug=True)