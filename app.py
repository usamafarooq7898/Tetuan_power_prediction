import tensorflow as tf
# FINAL CRITICAL FIXES: Import Keras metrics/models explicitly and correctly
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError 
from tensorflow.keras import models 

import joblib
from flask import Flask, request, jsonify
import numpy as np
import os 
import traceback 

# Initialize the Flask application
app = Flask(__name__)

# --- Initialize variables globally ---
model = None
scaler = None

# --- File paths ---
MODEL_PATH = 'lstm_power_prediction_model.h5'
SCALER_PATH = 'Tetuan_power_prediction_scaler.pkl'

# --- Load the Model and Scaler ---
try:
    # 1. CHECK FILE EXISTENCE
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
        
    # 2. LOAD WITH CUSTOM OBJECTS (THE FINAL, CORRECTED FIX)
    custom_objects = {
        # Using the directly imported class names without the 'metrics.' prefix
        'mse': MeanSquaredError(), 
        'mae': MeanAbsoluteError() 
    }

    model = models.load_model(
        MODEL_PATH, 
        custom_objects=custom_objects
    )
    scaler = joblib.load(SCALER_PATH)
    
    print("Model and Scaler loaded successfully!")

except Exception as e:
    # CRITICAL: Print the full stack trace for diagnostics
    print("--- MODEL LOADING FAILED: BEGIN TRACEBACK ---")
    traceback.print_exc()
    print(f"Error loading files: {type(e).__name__}: {e}")
    print("--- MODEL LOADING FAILED: END TRACEBACK ---")

# --- Define the API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    # CRITICAL SAFETY CHECK: Ensures the API returns a meaningful error if load failed
    if model is None or scaler is None:
        return jsonify({
            'error': 'Server initialization failed. Model/Scaler files were not loaded.',
            'details': 'Check the latest Render server logs for errors.'
        }), 500
    
    try:
        # 1. Get data from the POST request (expected as JSON)
        data = request.get_json(force=True)

        # 2. Extract and format the input features 
        input_data = np.array(data['features']).reshape(-1, 1) 

        # 3. Apply the same scaler used during training
        scaled_data = scaler.transform(input_data)

        # 4. Reshape for the LSTM model
        timesteps = 24 
        lstm_input = scaled_data.reshape(1, timesteps, 1) 

        # 5. Make prediction
        prediction_scaled = model.predict(lstm_input)

        # 6. Inverse transform the output
        prediction_actual = scaler.inverse_transform(prediction_scaled) 

        # 7. Return the prediction as a JSON response
        return jsonify({
            'prediction': prediction_actual.flatten().tolist()[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- Local testing hook ---
if __name__ == '__main__':
    app.run(debug=True)
