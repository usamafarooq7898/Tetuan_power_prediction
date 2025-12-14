import os
import numpy as np
import tensorflow as tf
import joblib
import traceback
import logging
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# --- CONFIGURATION AND LOGGING ---
logging.basicConfig(level=logging.INFO)

# Initialize the Flask application
app = Flask(__name__)

# --- Initialize variables globally ---
model = None
scaler = None

# --- File paths & Constants ---
# Use the correct file path for the model you wish to deploy (CNN was superior)
MODEL_PATH = 'cnn_power_prediction_model.h5' 
SCALER_PATH = 'Tetuan_power_prediction_scaler.pkl'

# These constants MUST match your training data/model setup
TIMESTEPS = 24       # Window size
N_FEATURES = 8       # 5 weather + 3 power zones (Input features)
N_TARGETS = 3        # 3 power zones (Output targets)

# --- Load the Model and Scaler ---
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
        
    # FIX: We remove 'custom_objects' to resolve the Keras serialization error.
    # Keras load_model should find the standard loss/metrics (mse, mae) internally.
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    logging.info("Model and Scaler loaded successfully!")

except Exception as e:
    logging.error("--- MODEL LOADING FAILED: BEGIN TRACEBACK ---")
    traceback.print_exc()
    logging.error(f"Error loading files: {type(e).__name__}: {e}")
    logging.error("--- MODEL LOADING FAILED: END TRACEBACK ---")

# --- Helper Function for Inverse Transformation (Crucial for 8-column scaler) ---
def inverse_transform_targets(y_scaled):
    """
    Takes scaled targets (N, 3) and inverse-transforms them using the 8-column scaler.
    """
    # Create an 8-column dummy array
    dummy = np.zeros((y_scaled.shape[0], N_FEATURES)) 
    # Place the 3 predictions into the last 3 columns
    dummy[:, -N_TARGETS:] = y_scaled 
    # Inverse transform the dummy array
    inv = scaler.inverse_transform(dummy)
    # Return only the 3 inverse-transformed target values
    return inv[:, -N_TARGETS:] 

# --- Define the API Endpoints ---

# FIX 1: Add the necessary root endpoint to resolve the 404 error
@app.route('/')
def index():
    return jsonify({
        'status': 'Service is online',
        'message': 'Use the /predict endpoint via POST to get forecasts.',
        'model_name': MODEL_PATH
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Server initialization failed. Model/Scaler not loaded.'}), 500
        
    try:
        data = request.get_json(force=True)
        # We expect 'data' to contain the 24x8 sequence
        sequence = data.get('data') 
        
        if not sequence:
            return jsonify({"error": "Missing 'data' sequence in request body."}), 400

        # FIX 2: Check input shape (must be 24x8)
        input_array = np.array(sequence, dtype=np.float32)
        if input_array.shape != (TIMESTEPS, N_FEATURES):
            return jsonify({
                "error": "Input shape mismatch.",
                "expected": f"A sequence of {TIMESTEPS} timesteps, each with {N_FEATURES} features: ({TIMESTEPS}, {N_FEATURES}).",
                "received": str(input_array.shape)
            }), 400

        # 1. Reshape and Scale Input
        # Scaling must be done on the 8-column array
        scaled_input = scaler.transform(input_array)
        
        # 2. Reshape for the model: (1_sample, 24_timesteps, 8_features)
        model_input = scaled_input[np.newaxis, :, :] 
        
        # 3. Make prediction: Output shape is (1, 3)
        # [0] to get the (3,) array of predictions
        prediction_scaled = model.predict(model_input)[0] 

        # 4. Inverse transform the 3 outputs using the helper function
        # Pass a (1, 3) array to the helper function
        prediction_actual = inverse_transform_targets(prediction_scaled[np.newaxis, :])
        
        # 5. Format results
        zone_predictions = {
            "Zone 1": float(prediction_actual[0][0]),
            "Zone 2": float(prediction_actual[0][1]),
            "Zone 3": float(prediction_actual[0][2])
        }

        return jsonify({
            'predictions': zone_predictions,
            'unit': 'kW',
            'input_shape_used': str(model_input.shape)
        })

    except Exception as e:
        logging.error("--- PREDICTION FAILED: BEGIN TRACEBACK ---")
        traceback.print_exc()
        logging.error(f"Error during prediction: {type(e).__name__}: {e}")
        logging.error("--- PREDICTION FAILED: END TRACEBACK ---")
        
        return jsonify({
            'error': 'Prediction processing failed.', 
            'details': 'Internal server error. Check logs for trace.',
            'specific_error': str(e)
        }), 500

# --- Local testing hook ---
if __name__ == '__main__':
    # Use the PORT environment variable for Render deployment compatibility
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
