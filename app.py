import os
import threading
import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
CORS(app)

# --- BACK4APP PARSE DATABASE CONFIGURATION (100% INTEGRATED) ---
PARSE_APP_ID = "Fwk5RhYZcVXho4Te49jOrwpkPa5KopFcpgEuWgS4"
PARSE_REST_KEY = "FTNByPOFkvid0a6jid0lFjeVP1dfiKjoouFse9JU"
PARSE_URL = "https://parseapi.back4app.com/classes/Prediction"

def _async_post_to_back4app(payload):
    """Background task to push database logs without blocking image uploads."""
    headers = {
        "X-Parse-Application-Id": PARSE_APP_ID,
        "X-Parse-REST-API-Key": PARSE_REST_KEY,
        "X-Parse-Client-Key": PARSE_REST_KEY,
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(PARSE_URL, json=payload, headers=headers, timeout=8)
        if response.status_code in [200, 201]:
            print("[Back4App DB Log] Record saved successfully in Prediction class.")
        else:
            print(f"[Back4App DB Warning] HTTP Status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[Back4App DB Error] Failed to reach Parse Database: {str(e)}")

def save_to_back4app_db(disease_name, confidence, severity_stage, farmer_advice):
    """Triggers asynchronous non-blocking thread for database writes."""
    payload = {
        "disease_name": disease_name,
        "confidence": confidence,
        "severity_stage": severity_stage,
        "farmer_advice": farmer_advice
    }
    threading.Thread(target=_async_post_to_back4app, args=(payload,)).start()

# 1. RENDER / BACK4APP CLOUD COMPATIBLE PATH LOGIC
MODEL_FILENAME = "tomato_disease_model.onnx"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

if os.path.exists(MODEL_PATH):
    print(f"Initializing Production Engine: Loading ONNX model structure from {MODEL_PATH}...")
    session = ort.InferenceSession(MODEL_PATH)
else:
    print(f"Warning: Model not found at path context {MODEL_PATH}. Checking working directory...")
    if os.path.exists(MODEL_FILENAME):
        session = ort.InferenceSession(MODEL_FILENAME)
    else:
        raise FileNotFoundError(
            f"Critical Deployment Error: '{MODEL_FILENAME}' was not found inside the root package context."
        )

input_name = session.get_inputs()[0].name

class_names = [
    'Target_Spot', 
    'Tomato___Early_blight', 
    'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 
    'healthy'
]

# --- AGRONOMIC ADVICE KNOWLEDGE BASE ---
ADVICE_DATABASE = {
    'Target_Spot': {
        'Early': "Prune lower infected leaves immediately to improve airflow. Avoid overhead watering.",
        'Medium': "Apply copper-based fungicides or chlorothalonil. Remove severely spotted leaves from the farm.",
        'Critical': "Disease has heavily spread. Apply systemic fungicides immediately. Clear and burn highly destroyed crops post-harvest."
    },
    'Tomato___Early_blight': {
        'Early': "Pluck off the bottom 2-3 infected leaves. Apply organic mulch around the stem base to prevent soil splashing.",
        'Medium': "Apply protective copper fungicides. Ensure proper crop spacing to reduce canopy humidity.",
        'Critical': "Infection is severe. Use powerful systemic fungicides (e.g., Mancozeb or Azoxystrobin). Avoid working in the field when plants are wet."
    },
    'Tomato___Late_blight': {
        'Early': "Highly contagious! Destroy infected leaflets immediately. Reduce field moisture and monitor closely.",
        'Medium': "Immediate chemical intervention required. Spray specialized late-blight fungicides containing Ridomil or copper compounds.",
        'Critical': "CRITICAL STATE! The spores will rapidly destroy the remaining crop. Uproot and burn heavily affected plants. Do not compost them."
    },
    'Tomato___Leaf_Mold': {
        'Early': "Increase ventilation in the greenhouse or field. Prune overcrowded branches to allow direct sunlight.",
        'Medium': "Lower greenhouse humidity below 85%. Spray preventative fungicides such as chlorothalonil.",
        'Critical': "Widespread leaf destruction. Apply highly active systemic fungicides and sanitize all pruning tools between plants."
    },
    'Tomato___Septoria_leaf_spot': {
        'Early': "Remove lower infected foliage. Ensure you water the base of the plant (drip irrigation), not the leaves.",
        'Medium': "Apply copper or potassium bicarbonate sprays. Keep the farm strictly free of weeds to reduce host vectors.",
        'Critical': "Severe defoliation risk. Spray intensive chemical fungicides every 7-10 days. Practice strict 3-year crop rotation next season."
    },
    'healthy': {
        'Safe': "Excellent farm management! Keep up regular scouting, maintain consistent drip irrigation, and apply balanced organic nutrition."
    }
}

# --- KEEP-AWAKE SELF PING ALGORITHM ---
def keep_server_awake():
    """Background task to query local deployment route preventing spin down."""
    try:
        self_url = os.environ.get('RENDER_EXTERNAL_URL') or os.environ.get('CONTAINER_URL')
        if self_url:
            ping_target = f"{self_url.rstrip('/')}/ping"
            print(f"[Keep-Awake] Pinging external routing gateway: {ping_target}")
            response = requests.get(ping_target, timeout=10)
            print(f"[Keep-Awake] Heartbeat acknowledged. Status Code: {response.status_code}")
        else:
            print("[Keep-Awake] External URL context not set. Pinging local internal loop...")
            port = int(os.environ.get("PORT", 8080))
            requests.get(f"http://127.0.0.1:{port}/ping", timeout=5)
    except Exception as e:
        print(f"[Keep-Awake Warning] System loop heartbeat skip: {str(e)}")

scheduler = BackgroundScheduler()
scheduler.add_job(func=keep_server_awake, trigger="interval", minutes=10)
scheduler.start()

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'active', 'message': 'Server pipeline keep-awake packet received successfully.'}), 200

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file element detected inside the request payload'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No physical image asset selected for diagnostic pipeline processing'}), 400

        # Structural Image Preprocessing
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))
        
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ONNX Inference Engine Parsing Pipeline Execution
        outputs = session.run(None, {input_name: img_array})
        predictions = outputs[0][0]  
        
        max_idx = np.argmax(predictions)
        result_class = class_names[max_idx]
        confidence_score = float(predictions[max_idx]) * 100

        # Dynamic Severity Stage Logic
        if result_class == 'healthy':
            severity_stage = "Safe"
        else:
            if confidence_score < 70.0:
                severity_stage = "Early Stage"
                lookup_stage = "Early"
            elif 70.0 <= confidence_score < 90.0:
                severity_stage = "Medium Stage"
                lookup_stage = "Medium"
            else:
                severity_stage = "Critical Stage"
                lookup_stage = "Critical"

        # Fetch Automated Farmer Advice
        if result_class == 'healthy':
            farmer_advice = ADVICE_DATABASE['healthy']['Safe']
        else:
            farmer_advice = ADVICE_DATABASE[result_class][lookup_stage]

        breakdown = {}
        for idx, name in enumerate(class_names):
            breakdown[name] = round(float(predictions[idx]) * 100, 2)

        clean_disease_name = result_class.replace('Tomato___', '').replace('_', ' ')
        formatted_confidence = f"{confidence_score:.2f}%"

        # SAVE TO BACK4APP PARSE DATABASE (ASYNC)
        save_to_back4app_db(
            disease_name=clean_disease_name,
            confidence=formatted_confidence,
            severity_stage=severity_stage,
            farmer_advice=farmer_advice
        )

        return jsonify({
            'status': 'success',
            'prediction': result_class,
            'clean_prediction': clean_disease_name,
            'confidence': formatted_confidence,
            'confidence_raw': round(confidence_score, 2),
            'severity_stage': severity_stage,
            'farmer_advice': farmer_advice,
            'breakdown': breakdown
        })

    except Exception as e:
        return jsonify({'error': f"Inference Failure Tracking: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    print(f"AI Tomato Mobile Gateway processing actively inside host engine port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
