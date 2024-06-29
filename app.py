from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the model
model = joblib.load('best_model.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = float(request.form.get("preg"))
        glu = float(request.form.get("glu"))
        bp = float(request.form.get("bp"))
        sk = float(request.form.get("sk"))
        ins = float(request.form.get("ins"))
        bmi = float(request.form.get("bmi"))
        dpf = float(request.form.get("dpf"))
        age = float(request.form.get("age"))
        
        input_features = np.array([preg, glu, bp, sk, ins, bmi, dpf, age]).reshape(1, -1)
        result = model.predict(input_features)
        
        if result[0] == 1:
            prediction = 'Suffered From diabetes.'
        else:
            prediction = 'Not Suffered From diabetes.'
        
        return jsonify(prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)