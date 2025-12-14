import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify # Ensure you have these imports

# --- START OF FILE-SPECIFIC IMPORTS AND SETUP ---
# CRITICAL: Define your target columns globally, as they are used for padding
TARGET_COLS = [
    "Zone 1 Power Consumption",
    "Zone 2 Power Consumption",
    "Zone 3 Power Consumption"
]

# Initialize Flask app
app = Flask(__name__)

# Load your models (Place your model loading code here)
try:
    # <--- YOUR EXISTING CODE HERE for loading the LSTM and CNN models --->
    # Example:
    lstm_model = tf.keras.models.load_model('lstm_model.h5')
    cnn_model = tf.keras.models.load_model('cnn_model.h5')
    
    # Load your scaler (if you have one)
    # Example:
    # scaler = load_scaler_function('scaler.pkl') 
    
    # Optional: Initial health check route
    @app.route('/', methods=['GET'])
    def home():
        return "Service is running and models are loaded."
    
except Exception as e:
    print(f"ERROR: Could not load models. Check file paths. Error: {e}")
    # Handle model loading failure gracefully if needed
    
# --- END OF FILE-SPECIFIC IMPORTS AND SETUP ---


@app.route('/predict', methods=['POST'])
def predict():
    # 1. Handle initial request and validation
    try:
        json_data = request.json
        if not isinstance(json_data, list):
            # The 'return' statement is correctly INDENTED here
            return jsonify({"status": "error", "message": "Input must be a list of 24 time steps."}), 400
    except Exception as e:
        # The 'return' statement is correctly INDENTED here
        return jsonify({"status": "error", "message": f"Invalid JSON input format: {e}"}), 400

    # 2. IMPLEMENT THE PADDING FIX (To handle 5-feature input for 8-feature model)
    # The 3 features that your lecturer will not send, but your 8-feature model needs.
    MISSING_LOAD_COLUMNS = TARGET_COLS 
    
    padded_data = []

    for time_step in json_data:
        # setdefault adds the key and sets the value to 0.0 only if the key is missing.
        time_step.setdefault(MISSING_LOAD_COLUMNS[0], 0.0)
        time_step.setdefault(MISSING_LOAD_COLUMNS[1], 0.0)
        time_step.setdefault(MISSING_LOAD_COLUMNS[2], 0.0)
        
        padded_data.append(time_step)
        
    # 3. Data Preprocessing and Prediction
    try:
        # a. Convert padded data (now 8 features) to a DataFrame
        data_df = pd.DataFrame(padded_data)
        
        # b. Ensure column order matches your training data (CRITICAL!)
        # <--- YOUR EXISTING CODE HERE to select and reorder the 8 feature columns --->
        # Example (Ensure the 8 columns are in the EXACT order your model expects):
        # feature_order = ["Temperature", "Humidity", "Wind Speed", "general diffuse flows", 
        #                  "diffuse flows", "Zone 1 Power Consumption", "Zone 2 Power Consumption", 
        #                  "Zone 3 Power Consumption"]
        # data_df = data_df[feature_order]
        
        data_array = data_df.values
        
        # c. Reshape and Scale data
        # <--- YOUR EXISTING CODE HERE for scaling and reshaping the data_array (e.g., to 1, 24, 8) --->
        # Example:
        # scaled_data = scaler.transform(data_array) 
        # reshaped_data = scaled_data.reshape(1, 24, 8) 
        
        # d. Make Predictions
        # <--- YOUR EXISTING CODE HERE for LSTM and CNN prediction --->
        # Example:
        # lstm_pred_scaled = lstm_model.predict(reshaped_data)[0][0]
        # cnn_pred_scaled = cnn_model.predict(reshaped_data)[0][0]
        
        # e. Inverse Transform the Prediction (if you used a scaler)
        # <--- YOUR EXISTING CODE HERE for inverse transforming the predictions --->
        # Example:
        # # Create a placeholder array for inverse transform
        # placeholder = np.zeros((1, 8)) 
        # placeholder[0, 5] = lstm_pred_scaled # Put prediction in the Zone 1 column slot
        # lstm_pred = scaler.inverse_transform(placeholder)[0, 5]
        # # Repeat for other zones/models...
        
        # f. Ensemble and final result compilation
        # <--- YOUR EXISTING CODE HERE for ensemble prediction and final dictionary --->
        # Example:
        final_result = {
            "status": "success",
            "prediction_zone_1": 15000.0, # Replace with calculated inverse transformed values
            "prediction_zone_2": 18000.0,
            "prediction_zone_3": 20000.0
        }
        
    except Exception as e:
        # This will catch errors during padding, scaling, or prediction
        return jsonify({"status": "error", "message": f"Processing or Prediction Error: {e}"}), 500

    # 4. Return the Final Prediction
    return jsonify(final_result)


# --- This line should be at the very end of your app
