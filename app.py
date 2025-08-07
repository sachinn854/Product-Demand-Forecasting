from flask import Flask, render_template, request
from predict import load_model, predict_demand

app = Flask(__name__)
model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html', error="Model not loaded.")

    try:
        form_data = request.form.to_dict()
        prediction, insights = predict_demand(model, form_data)
        return render_template('result.html', prediction=prediction, insights=insights)
    except Exception as e:
        return render_template('result.html', error=f"Prediction failed: {e}")

if __name__ == '__main__':
    app.run(debug=True)