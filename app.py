from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('credit_model.pkl')
features = joblib.load('feature_names.pkl')

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = []
    for feature in features:
        value = float(request.form[feature])
        input_data.append(value)
    
    prediction = model.predict([input_data])[0]
    result = "High Risk of Default" if prediction == 1 else "Likely Creditworthy"
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
