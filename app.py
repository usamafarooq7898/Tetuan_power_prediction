# Insert this entire block inside your @app.route('/predict', methods=['POST']) function:

# 1. Define the missing columns (the three past load values)
MISSING_LOAD_COLUMNS = [
    "Zone 1 Power Consumption",
    "Zone 2 Power Consumption",
    "Zone 3 Power Consumption"
]

# 2. Get the incoming data (which only has 5 features per timestep)
try:
    json_data = request.json
    if not isinstance(json_data, list):
        return jsonify({"status": "error", "message": "Input must be a list of 24 time steps."}), 400
except Exception:
    return jsonify({"status": "error", "message": "Invalid JSON input format."}), 400

# 3. Apply the Padding/Imputation Fix
# We iterate through each of the 24 time steps (dictionaries) and add the 
# missing keys with a default value of 0.0 (Zero Imputation).
padded_data = []

for time_step in json_data:
    # setdefault adds the key and sets the value to 0.0 only if the key is missing.
    time_step.setdefault(MISSING_LOAD_COLUMNS[0], 0.0)
    time_step.setdefault(MISSING_LOAD_COLUMNS[1], 0.0)
    time_step.setdefault(MISSING_LOAD_COLUMNS[2], 0.0)
    
    # The dictionary 'time_step' now correctly has 8 keys (5 original + 3 padded)
    padded_data.append(time_step)

# 4. CRITICAL STEP: Now, your original code that converts data to 
# a DataFrame/NumPy array MUST use 'padded_data' instead of 'request.json' 
# or the original 'json_data' variable.

# Example of what comes next in your app.py:
# data_df = pd.DataFrame(padded_data) 
# data_array = data_df.values
# ... (feed data_array to your model)
