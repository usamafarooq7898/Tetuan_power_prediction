import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# --- Configuration ---
# Set the port for gunicorn
PORT = int(os.environ.get("PORT", 10000))

# The actual loading of these variables is now handled by the separate init_app.py script
lstm_model = None
cnn_model = None
scaler = None

# --- Re-Initialize Models and Scaler (Must be loaded when app starts) ---
try:
    # We must load the models and scaler here as well, as they are not global between processes
    MODEL_PATH_LSTM = 'lstm_power_prediction_model.h5'
    MODEL_PATH_CNN = 'cnn_power_prediction_model.h5'
    DATA_PATH = 'Tetuan City power consumption.xlsx - in.csv'
    
    # Use the simple load_model, as the complex startup is now handled by init_app.py
    lstm_model = load_model(MODEL_PATH_LSTM)
    cnn_model = load_model(MODEL_PATH_CNN)
    
    # Reload data and fit scaler (each Gunicorn worker needs its own copy)
    df = pd.read_csv(DATA_PATH)
    feature_cols = df.columns[1:9]
    data_for_scaler = df[feature_cols].values
    
    scaler = StandardScaler()
    scaler.fit(data_for_scaler)
    
except Exception as e:
    print(f"ERROR: Worker failed to load models/scaler. This is expected if init_app.py failed, but should be checked if not. Error: {e}")

# --- Flask App Initialization ---
app = Flask(__name__)
TIME_STEPS = 24  # Lookback period for LSTM/CNN

# --- Helper Function for Prediction ---
def prepare_data_for_prediction(data_point):
    """
    Prepares a single 24-hour block of data for model input.
    """
    try:
        # Convert list of dicts to DataFrame
        df_input = pd.DataFrame(data_point)
        
        # Select the same 8 feature columns used for training/scaling
        feature_cols = [
            "Temperature", "Humidity", "Wind Speed", 
            "general diffuse flows", "diffuse flows", 
            "Zone 1 Power Consumption", "Zone 2 Power Consumption", 
            "Zone 3 Power Consumption"
        ]
        
        input_data = df_input[feature_cols].values
        
        # Scale the data
        input_data_scaled = scaler.transform(input_data)
        
        # Reshape to (1, TIME_STEPS, n_features) -> (1, 24, 8)
        X = input_data_scaled.reshape(1, TIME_STEPS, len(feature_cols))
        
        return X
    
    except Exception as e:
        app.logger.error(f"Data preparation failed: {e}")
        return None

# --- API Routes ---

@app.route("/", methods=["GET"])
def home():
    """Simple status check for the service."""
    if lstm_model and cnn_model and scaler:
        return "Service is running and models are loaded.", 200
    else:
        # If models failed to load in the worker process, return 500
        return "Service is running, but models failed to load in the worker process. Check logs.", 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives 24 hours of data and returns a 1-hour ahead prediction.
    """
    
    if not lstm_model or not cnn_model or not scaler:
        return jsonify({"error": "Models or Scaler not loaded in worker. Cannot predict."}), 500

    # Expect JSON data
    data = request.get_json(force=True)
    
    # Validation: Check if the input is a list of 24 data points
    if not isinstance(data, list) or len(data) != TIME_STEPS:
        return jsonify({
            "error": f"Invalid input format. Expected a list of exactly {TIME_STEPS} data points (dictionaries)."
        }), 400

    # 1. Prepare data
    X = prepare_data_for_prediction(data)
    if X is None:
        return jsonify({"error": "Failed to process input data for prediction."}), 400

    # 2. Make predictions
    try:
        # The code must use the model objects loaded in the worker process
        lstm_prediction_scaled = lstm_model.predict(X)
        cnn_prediction_scaled = cnn_model.predict(X)
        
        # 3. Inverse transform the prediction
        
        # Assuming the model is predicting 'Zone 1 Power Consumption' (the 6th column, index 5)
        # Create a dummy array (1 row, 8 columns) and place the prediction in the correct spot (index 5)
        dummy_array_lstm = np.zeros((1, scaler.n_features_in_))
        dummy_array_lstm[0, 5] = lstm_prediction_scaled[0, 0]
        
        dummy_array_cnn = np.zeros((1, scaler.n_features_in_))
        dummy_array_cnn[0, 5] = cnn_prediction_scaled[0, 0]
        
        # Inverse transform
        lstm_prediction_inverse = scaler.inverse_transform(dummy_array_lstm)[0, 5]
        cnn_prediction_inverse = scaler.inverse_transform(dummy_array_cnn)[0, 5]
        
        # Calculate the ensemble (average) prediction
        ensemble_prediction = (lstm_prediction_inverse + cnn_prediction_inverse) / 2
        
        # 4. Return results
        return jsonify({
            "status": "success",
            "lstm_prediction": round(float(lstm_prediction_inverse), 2
