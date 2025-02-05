from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("mnist_model.h5")

app = Flask(__name__)

@app.route('/')
def home():
    return "MNIST Digit Recognition API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json['data']
        data = np.array(data).reshape(1, 28, 28) / 255.0  # Normalize input
        
        # Make prediction
        prediction = model.predict(data)
        predicted_digit = np.argmax(prediction)

        return jsonify({"prediction": int(predicted_digit)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
