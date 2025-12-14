import pandas as pd
# Change import to use TensorFlow's Keras backend explicitly
from tensorflow.keras.models import load_model
# Import CustomObjectScope from tf.keras.utils
from tensorflow.keras.utils import CustomObjectScope 

MODEL_PATH_LSTM = 'lstm_power_prediction_model.h5'
MODEL_PATH_CNN = 'cnn_power_prediction_model.h5'
DATA_PATH = 'tetuan_power_consumption_data.csv' 

# This script runs FIRST and verifies that all files exist and are loadable.
try:
    print("INFO: --- STARTING INITIALIZATION CHECK ---")
    
    # 1. Verify data file exists and can be read
    df = pd.read_csv(DATA_PATH, encoding='latin-1', sep=r'\s+', index_col=0)
    print("INFO: Data file loaded successfully.")

    # 2. Verify models exist and can be loaded
    # FIX 2: Using CustomObjectScope from tf.keras.utils 
    # and importing load_model from tf.keras to resolve version conflicts.
    with CustomObjectScope({}):
        load_model(MODEL_PATH_LSTM, compile=False)
        load_model(MODEL_PATH_CNN, compile=False)
    
    print("INFO: All models loaded successfully.")

except Exception as e:
    # If this fails, the deployment stops immediately.
    print(f"FATAL ERROR: Initialization check failed before server start. The models or data file could not be read. Error: {e}")
    exit(1)

print("INFO: Initialization complete. Proceeding to run Gunicorn.")
