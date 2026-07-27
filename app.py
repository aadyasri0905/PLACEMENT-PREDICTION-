from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])

    data = np.array([[cgpa, iq]])

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "🎉 Congratulations! You are likely to get placed."
    else:
        result = "😔 Sorry! Based on the model, placement chances are low."

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)