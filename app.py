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
MODEL_PATH_LSTM = 'lstm_power_prediction_model.h5'
MODEL_PATH_CNN = 'cnn_power_prediction_model.h5'
DATA_PATH = 'Tetuan City power consumption.xlsx - in.csv'
TIME_STEPS = 24  # Lookback period for LSTM/CNN

# --- Model Loading and Caching ---
lstm_model = None
cnn_model = None
scaler = None

# Custom error handling for model loading
def safe_load_model(path):
    """Loads a Keras model, ignoring potentially mismatched serialization arguments."""
    try:
        # safe_mode=False ignores unrecognized arguments like 'batch_shape' 
        # which fixes the deserialization error between different TF versions.
        model = load_model(path, safe_mode=False)
        print(f"INFO: Successfully loaded model from {path}")
        return model
    except Exception as e:
        print(f"ERROR:--- MODEL LOADING FAILED: BEGIN TRACEBACK ---")
        print(f"Traceback (most recent call last):")
        # Print only the core error message for clarity
        print(f"Error loading files: {type(e).__name__}: {e}")
        print(f"ERROR:--- MODEL LOADING FAILED: END TRACEBACK ---")
        return None

# Load models and scaler when the application starts
try:
    lstm_model = safe_load_model(MODEL_PATH_LSTM)
    cnn_model = safe_load_model(MODEL_PATH_CNN)
    
    # Load data for scaler fitting (assuming this data file is in the root directory)
    df = pd.read_csv(DATA_PATH)
    
    # Select feature columns (8 columns needed for your model input shape [24, 8])
    feature_cols = df.columns[1:9]
    data_for_scaler = df[feature_cols].values
    
    scaler = StandardScaler()
    scaler.fit(data_for_scaler)
    
except FileNotFoundError:
    print(f"ERROR: Required files not found. Check if {MODEL_PATH_LSTM}, {MODEL_PATH_CNN}, and {DATA_PATH} exist.")
except Exception as e:
    print(f"ERROR: Error during initialization: {e}")

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Helper Function for Prediction ---
def prepare_data_for_prediction(data_point):
    """
    Prepares a single 24-hour block of data for model input.
    The input data_point must be a list of 24 dictionaries.
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
        return "Service is running, but models failed to load. Check logs.", 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives 24 hours of data and returns a 1-hour ahead prediction.
    """
    
    if not lstm_model or not cnn_model or not scaler:
        return jsonify({"error": "Models or Scaler not loaded. Cannot predict."}), 500

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
        lstm_prediction_scaled = lstm_model.predict(X)
        cnn_prediction_scaled = cnn_model.predict(X)
        
        # 3. Inverse transform the prediction
        
        # The scaler was fitted on 8 features. To inverse transform a single value (the prediction),
        # we need to put it back into an array of 8 columns.
        
        # For this example, we'll assume the model is predicting 'Zone 1 Power Consumption' 
        # (the 6th column, index 5)
        
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
            "lstm_prediction": round(float(lstm_prediction_inverse), 2),
            "cnn_prediction": round(float(cnn_prediction_inverse), 2),
            "ensemble_prediction": round(float(ensemble_prediction), 2)
        })

    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500


# Run the app if executed directly (e.g., for local testing, though gunicorn is used on Render)
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT, debug=True)
