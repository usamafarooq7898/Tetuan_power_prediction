import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- Load the Model and Scaler ---
# It's crucial to load these outside the prediction function so they only load once
model=None
scaler=None
try:
# Define custom objects to handle the Keras metric (e.g., if you used custom MSE/MAE)
custom_objects = {
    'mse': tf.keras.metrics.mean_squared_error,
    'mae': tf.keras.metrics.mean_absolute_error 
    # Add any other custom functions or metrics you used in your model training here
}

model = tf.keras.models.load_model(
    'lstm_power_prediction_model.h5', 
    custom_objects=custom_objects
)    scaler = joblib.load('Tetuan_power_prediction_scaler.pkl')
    print("Model and Scaler loaded successfully!")
except Exception as e:
    # If deployment fails, this error helps you debug why the files weren't found/loaded
    print(f"Error loading files: {e}")

# --- Define the API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from the POST request (expected as JSON)
        data = request.get_json(force=True)

        # 2. Extract and format the input features 
        # The 'features' key must match what the client sends in the JSON body.
        # Example JSON: {"features": [10.5, 20.1, ..., 9.8]} (24 values for 24 timesteps)

        # The input data is expected to be a flat list of 24 values
        input_data = np.array(data['features']).reshape(-1, 1) # Reshape to (24, 1)

        # 3. Apply the same scaler used during training
        scaled_data = scaler.transform(input_data)

        # 4. Reshape for the LSTM model (TensorFlow expects [samples, timesteps, features])
        # Assuming your model was trained on 24 timesteps and 1 feature:
        timesteps = 24 

        # Reshape the (24, 1) array into (1, 24, 1)
        lstm_input = scaled_data.reshape(1, timesteps, 1) 

        # 5. Make prediction
        prediction_scaled = model.predict(lstm_input)

        # 6. Inverse transform the output to get the actual power value
        # Since the model predicts a single value, we inverse transform that single value
        prediction_actual = scaler.inverse_transform(prediction_scaled) 

        # 7. Return the prediction as a JSON response
        return jsonify({
            'prediction': prediction_actual.flatten().tolist()[0] # Returns a single value
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- The hosting service (Gunicorn) will ignore this, but it's good for local testing ---
if __name__ == '__main__':
    app.run(debug=True)
