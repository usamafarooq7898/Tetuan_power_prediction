import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

MODEL_PATH_LSTM = 'lstm_power_prediction_model.h5'
MODEL_PATH_CNN = 'cnn_power_prediction_model.h5'
DATA_PATH = 'Tetuan City power consumption.xlsx - in.csv'

# This script runs FIRST and verifies that all files exist and are loadable.
try:
    print("INFO: --- STARTING INITIALIZATION CHECK ---")
    
    # 1. Verify data file exists and can be read
    df = pd.read_csv(DATA_PATH)
    print("INFO: Data file loaded successfully.")

    # 2. Verify models exist and can be loaded
    # The loading here is purely a check. The models are loaded again in app.py for each worker.
    load_model(MODEL_PATH_LSTM)
    load_model(MODEL_PATH_CNN)
    print("INFO: All models loaded successfully.")

except Exception as e:
    print(f"FATAL ERROR: Initialization check failed before server start. The models or data file could not be read. Error: {e}")
    # Exit with a non-zero status to stop the deployment if setup fails
    exit(1)

print("INFO: Initialization complete. Proceeding to run Gunicorn.")