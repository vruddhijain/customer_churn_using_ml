import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/predict", methods = ["POST"])

def predict():
    float_feat = [float(x) for x in request.form.values()]
    feat = [np.array(float_feat)]
    prediction = model.predict(feat)

    return render_template("index.html", prediction_text = "the churn is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)