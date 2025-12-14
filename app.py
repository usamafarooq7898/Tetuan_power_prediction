import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# --- Configuration ---
PORT = int(os.environ.get("PORT", 10000))

# Global variables initialized to None
lstm_model = None
cnn_model = None
scaler = None

# --- File Paths ---
MODEL_PATH_LSTM = 'lstm_power_prediction_model.h5'
MODEL_PATH_CNN = 'cnn_power_prediction_model.h5'
DATA_PATH = 'tetuan_power_consumption_data.csv' 
TIME_STEPS = 24  # Lookback period for LSTM/CNN

# --- Worker Initialization (Executed when Gunicorn worker starts) ---
try:
    # Load models and scaler for the worker process
    lstm_model = load_model(MODEL_PATH_LSTM)
    cnn_model = load_model(MODEL_PATH_CNN)
    
    # Load data for scaler fitting (FINAL FIXES: encoding and separator)
    df = pd.read_csv(DATA_PATH, encoding='latin-1', sep=';')
    
    # Select feature columns (8 columns needed for your model input shape [24, 8])
    # Assuming columns 1 to 9 (Date/Time is 0, then 8 features)
    feature_cols = df.columns[1:9]
    data_for_scaler = df[feature_cols].values
    
    scaler = StandardScaler()
    scaler.fit(data_for_scaler)
    
except Exception as e:
    # This print helps debug if loading fails inside the Gunicorn worker
    print(f"ERROR: Worker failed to load models/scaler. Error: {e}")

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Helper Function for Prediction ---
def prepare_data_for_prediction(data_point):
    """
    Prepares a single 24-hour block of data for model input.
    """
    
    # Note: If you encounter key errors here, you might need to adjust feature_cols to 
    # match the exact column names after the semicolon separation is used.
    try:
        df_input = pd.DataFrame(data_point)
        
        feature_cols = [
            "Temperature", "Humidity", "Wind Speed", 
            "general diffuse flows", "diffuse flows", 
            "Zone 1 Power Consumption", "Zone 2 Power Consumption", 
            "Zone 3 Power Consumption"
        ]
        
        input_data = df_input[feature_cols].values
        
        # Scale the data using the fitted scaler
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
        # This occurs if the worker failed the try-except block above
        return "Service is running, but models failed to load in the worker process. Check logs.", 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives 24 hours of data and returns a 1-hour ahead prediction.
    """
    
    if not lstm_model or not cnn_model or not scaler:
        return jsonify({"error": "Models or Scaler not loaded in worker. Cannot predict."}), 500

    data = request.get_json(force=True)
    
    if not isinstance(data, list) or len(data) != TIME_STEPS:
        return jsonify({
            "error": f"Invalid input format. Expected a list of exactly {TIME_STEPS} data points (dictionaries)."
        }), 400

    X = prepare_data_for_prediction(data)
    if X is None:
        return jsonify({"error": "Failed to process input data for prediction."}), 400

    try:
        lstm_prediction_scaled = lstm_model.predict(X)
        cnn_prediction_scaled = cnn_model.predict(X)
        
        # Inverse transform the prediction (assuming Zone 1 Power Consumption - index 5)
        dummy_array_lstm = np.zeros((1, scaler.n_features_in_))
        dummy_array_lstm[0, 5] = lstm_prediction_scaled[0, 0]
        
        dummy_array_cnn = np.zeros((1, scaler.n_features_in_))
        dummy_array_cnn[0, 5] = cnn_prediction_scaled[0, 0]
        
        lstm_prediction_inverse = scaler.inverse_transform(dummy_array_lstm)[0, 5]
        cnn_prediction_inverse = scaler.inverse_transform(dummy_array_cnn)[0, 5]
        
        ensemble_prediction = (lstm_prediction_inverse + cnn_prediction_inverse) / 2
        
        return jsonify({
            "status": "success",
            "lstm_prediction": round(float(lstm_prediction_inverse), 2),
            "cnn_prediction": round(float(cnn_prediction_inverse), 2),
            "ensemble_prediction": round(float(ensemble_prediction), 2)
        })

    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT, debug=True)
