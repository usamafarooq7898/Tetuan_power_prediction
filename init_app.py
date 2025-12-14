import pandas as pd
from keras.models import load_model

MODEL_PATH_LSTM = 'lstm_power_prediction_model.h5'
MODEL_PATH_CNN = 'cnn_power_prediction_model.h5'
DATA_PATH = 'tetuan_power_consumption_data.csv' 

# This script runs FIRST and verifies that all files exist and are loadable.
try:
    print("INFO: --- STARTING INITIALIZATION CHECK ---")
    
    # 1. Verify data file exists and can be read (FINAL FINAL FIX: Regex separator and index column)
    df = pd.read_csv(DATA_PATH, encoding='latin-1', sep=r'\s+', index_col=0)
    print("INFO: Data file loaded successfully.")

    # 2. Verify models exist and can be loaded
    # FIX: Adding compile=False to bypass layer deserialization issues caused by version mismatch.
    load_model(MODEL_PATH_LSTM, compile=False)
    load_model(MODEL_PATH_CNN, compile=False)
    print("INFO: All models loaded successfully.")

except Exception as e:
    # If this fails, the deployment stops immediately.
    print(f"FATAL ERROR: Initialization check failed before server start. The models or data file could not be read. Error: {e}")
    exit(1)

print("INFO: Initialization complete. Proceeding to run Gunicorn.")
