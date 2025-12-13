import os
import numpy as np
import tensorflow as tf
import joblib
import traceback
import logging
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError # Import classes directly
from tensorflow.keras import models 

# --- CONFIGURATION AND LOGGING ---
logging.basicConfig(level=logging.INFO)

# Initialize the Flask application
app = Flask(__name__)

# --- Initialize variables globally ---
model = None
scaler = None

# --- File paths ---
MODEL_PATH = 'lstm_power_prediction_model.h5'
SCALER_PATH = 'Tetuan_power_prediction_scaler.pkl'
TIMESTEPS = 24 # Define the fixed timestep length

# --- Load the Model and Scaler ---
try:
    # 1. CHECK FILE EXISTENCE
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
        
    # 2. LOAD WITH CUSTOM OBJECTS (THE FINAL, CORRECTED FIX)
    # Use the class name itself as the key and the class reference as the value
    custom_objects = {
        'MeanSquaredError': MeanSquaredError,  
        'MeanAbsoluteError': MeanAbsoluteError 
    }

    model = load_model(
        MODEL_PATH, 
        custom_objects=custom_objects
    )
    # Ensure the correct 1-feature scaler is loaded!
    scaler = joblib.load(SCALER_PATH)
    
    logging.info("Model and Scaler loaded successfully!")

except Exception as e:
    # CRITICAL: Print the full stack trace for diagnostics
    logging.error("--- MODEL LOADING FAILED: BEGIN TRACEBACK ---")
    traceback.print_exc()
    logging.error(f"Error loading files: {type(e).__name__}: {e}")
    logging.error("--- MODEL LOADING FAILED: END TRACEBACK ---")

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
        features = data.get('features')
        
        if not features:
             return jsonify({"error": "Missing 'features' in request body."}), 400

        # 2. Convert input features and reshape for Scaler (2D: (N_timesteps, 1_feature))
        # Input: (24,) -> Output: (24, 1)
        input_array_2d = np.array(features, dtype=np.float32).reshape(-1, 1)

        # 3. Apply the same scaler used during training
        scaled_data = scaler.transform(input_array_2d)
        logging.info(f"Shape after scaling (2D): {scaled_data.shape}")

        # 4. Reshape for the LSTM model (3D: (1_sample, N_timesteps, 1_feature))
        # Input: (24, 1) -> Output: (1, 24, 1)
        # np.newaxis is the most reliable way to add the batch dimension.
        lstm_input = scaled_data[np.newaxis, :, :] 
        logging.info(f"Shape before model (3D): {lstm_input.shape}")

        # 5. Make prediction
        prediction_scaled = model.predict(lstm_input)

        # 6. Inverse transform the output (Prediction output is (1, 1))
        prediction_actual = scaler.inverse_transform(prediction_scaled) 

        # 7. Return the prediction as a JSON response
        # Extract the single float value from the (1, 1) array.
        final_prediction = float(prediction_actual[0][0])
        
        return jsonify({
            'prediction': final_prediction,
            'unit': 'kW',
            'input_shape_used': str(lstm_input.shape)
        })

    except Exception as e:
        # Use traceback logging for errors inside the predict route too!
        logging.error("--- PREDICTION FAILED: BEGIN TRACEBACK ---")
        traceback.print_exc()
        logging.error(f"Error during prediction: {type(e).__name__}: {e}")
        logging.error("--- PREDICTION FAILED: END TRACEBACK ---")
        
        return jsonify({
            'error': 'Prediction processing failed.', 
            'details': str(e)
        }), 500

# --- Local testing hook ---
# Gunicorn handles running this in Render, this block is for local testing only
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
