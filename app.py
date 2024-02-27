from flask import Flask, render_template, request
import numpy as np
from sklearn.svm import SVC


app = Flask(__name__)

# Define a function to train the SVM model
def train_model():
    # Synthetic data for demonstration purposes
    X = np.random.rand(100, 10)  # Replace with your own dataset
    y = np.random.randint(0, 2, 100)  # Replace with your own dataset labels

    # Train the SVM model
    model = SVC()
    model.fit(X, y)

    return model

# Load the trained SVM model
model = train_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get user input from the form
    age = float(request.form.get("age"))
    gender = float(request.form.get("gender"))
    tb = float(request.form.get("tb"))
    db = float(request.form.get("db"))
    alkphos = float(request.form.get("alkphos"))
    sgpt = float(request.form.get("sgpt"))
    sgot = float(request.form.get("sgot"))
    tp = float(request.form.get("tp"))
    alb = float(request.form.get("alb"))
    a_g_ratio = float(request.form.get("a_g_ratio"))

    # Create feature vector from user input
    features = np.array([[age, gender, tb, db, alkphos, sgpt, sgot, tp, alb, a_g_ratio]])

    # Use the trained SVM model to make predictions
    prediction = model.predict(features)

    # Map the prediction to the diagnosis result
    if prediction == 1:
        result = "You should consult a doctor. Liver disease is suspected."
    else:
        result = "You are healthy. No signs of liver disease."

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
