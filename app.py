import os
import threading
import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
# Ruhusu CORS kwenye njia zote kikamilifu
CORS(app, resources={r"/*": {"origins": "*"}})

# --- BACK4APP PARSE CONFIGURATION ---
PARSE_APP_ID = "Fwk5RhYZcVXho4Te49jOrwpkPa5KopFcpgEuWgS4"
PARSE_REST_KEY = "FTNByPOFkvid0a6jid0lFjeVP1dfiKjoouFse9JU"
PARSE_URL = "https://parseapi.back4app.com/classes/Prediction"

def log_to_back4app_background(payload):
    """Executes DB post safely without blocking or throwing unhandled HTTP errors."""
    try:
        headers = {
            "X-Parse-Application-Id": PARSE_APP_ID,
            "X-Parse-REST-API-Key": PARSE_REST_KEY,
            "Content-Type": "application/json"
        }
        res = requests.post(PARSE_URL, json=payload, headers=headers, timeout=4)
        print(f"[DB Log Status]: {res.status_code} - {res.text}")
    except Exception as err:
        print(f"[DB Async Exception]: {str(err)}")

# MODEL LOADING
MODEL_FILENAME = "tomato_disease_model.onnx"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

if os.path.exists(MODEL_PATH):
    session = ort.InferenceSession(MODEL_PATH)
elif os.path.exists(MODEL_FILENAME):
    session = ort.InferenceSession(MODEL_FILENAME)
else:
    raise FileNotFoundError("Model file not found on server context!")

input_name = session.get_inputs()[0].name
class_names = [
    'Target_Spot', 
    'Tomato___Early_blight', 
    'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 
    'healthy'
]

ADVICE_DATABASE = {
    'Target_Spot': {'Early': "Prune lower infected leaves.", 'Medium': "Apply copper-based fungicides.", 'Critical': "Apply systemic fungicides immediately."},
    'Tomato___Early_blight': {'Early': "Pluck off bottom leaves.", 'Medium': "Apply protective copper fungicides.", 'Critical': "Use systemic fungicides like Mancozeb."},
    'Tomato___Late_blight': {'Early': "Destroy infected leaflets.", 'Medium': "Spray specialized late-blight fungicides.", 'Critical': "Uproot heavily affected plants."},
    'Tomato___Leaf_Mold': {'Early': "Increase airflow.", 'Medium': "Spray chlorothalonil.", 'Critical': "Apply active systemic fungicides."},
    'Tomato___Septoria_leaf_spot': {'Early': "Remove lower infected foliage.", 'Medium': "Apply copper sprays.", 'Critical': "Spray intensive fungicides every 7 days."},
    'healthy': {'Safe': "Keep up regular scouting and clean farm management."}
}

@app.errorhandler(Exception)
def handle_global_exception(e):
    """Catches all errors and attaches CORS headers so browser shows real error message."""
    response = jsonify({"error": f"Internal Server Error: {str(e)}"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response, 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file element detected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # 1. Image Preprocessing & Inference
    img = Image.open(file).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(img).astype(np.float32) / 255.0, axis=0)

    outputs = session.run(None, {input_name: img_array})
    predictions = outputs[0][0]
    
    max_idx = np.argmax(predictions)
    result_class = class_names[max_idx]
    confidence_score = float(predictions[max_idx]) * 100

    # 2. Advice & Severity Logic
    if result_class == 'healthy':
        severity_stage = "Safe"
        farmer_advice = ADVICE_DATABASE['healthy']['Safe']
    else:
        if confidence_score < 70.0:
            severity_stage, lookup_stage = "Early Stage", "Early"
        elif 70.0 <= confidence_score < 90.0:
            severity_stage, lookup_stage = "Medium Stage", "Medium"
        else:
            severity_stage, lookup_stage = "Critical Stage", "Critical"
        farmer_advice = ADVICE_DATABASE[result_class][lookup_stage]

    clean_disease_name = result_class.replace('Tomato___', '').replace('_', ' ')
    formatted_confidence = f"{confidence_score:.2f}%"

    # 3. Fire-and-forget Database Logging (Daemon Thread)
    db_payload = {
        "disease_name": clean_disease_name,
        "confidence": formatted_confidence,
        "severity_stage": severity_stage,
        "farmer_advice": farmer_advice
    }
    t = threading.Thread(target=log_to_back4app_background, args=(db_payload,))
    t.daemon = True
    t.start()

    # 4. Return API Response
    return jsonify({
        'status': 'success',
        'prediction': result_class,
        'clean_prediction': clean_disease_name,
        'confidence': formatted_confidence,
        'severity_stage': severity_stage,
        'farmer_advice': farmer_advice
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
