import tensorflow as tf
# CRITICAL FIX: Explicitly import Keras metrics for deserialization compatibility
from tensorflow.keras import metrics 

import joblib
from flask import Flask, request, jsonify
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- Load the Model and Scaler ---
# Initialize globally to prevent 'is not defined' errors if loading fails
model = None
scaler = None

try:
    # CRITICAL FIX for Deserialization Error: Pass the actual metric function object
    # using the explicit Keras import (metrics)
    custom_objects = {
        'mse': metrics.mean_squared_error,
        'mae': metrics.mean_absolute_error 
        # Add any other custom functions or metrics you used in your model training here
    }

    model = tf.keras.models.load_model(
        'lstm_power_prediction_model.h5', 
        custom_objects=custom_objects
    )
    scaler = joblib.load('Tetuan_power_prediction_scaler.pkl')
    
    print("Model and Scaler loaded successfully!")

except Exception as e:
    # If deployment fails, this will now print the exact root cause
    print(f"Error loading files: {e}")

# --- Define the API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    # CRITICAL SAFETY CHECK: Ensures the API returns a meaningful error if load failed
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler failed to load during server startup. Check server logs in Render.'}), 500
    
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

# --- The hosting service (Gunicorn) will ignore this, but it's good for local testing ---
if __name__ == '__main__':
    app.run(debug=True)
