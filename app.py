from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_email(text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]
    
    return "FAKE / PHISHING EMAIL" if pred == 1 else "SAFE EMAIL"

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        email_text = request.form["email"]
        result = predict_email(email_text)

    return render_template("index.htm", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)