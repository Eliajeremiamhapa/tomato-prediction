import os
import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
CORS(app)

# 1. RENDER CLOUD COMPATIBLE PATH LOGIC
# Moves from static Windows paths to production relative environment layout
MODEL_FILENAME = "tomato_disease_model.onnx"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

# Validation layer ensuring your weights array is present inside the container filesystem
if os.path.exists(MODEL_PATH):
    print(f"Initializing Production Engine: Loading ONNX model structure from {MODEL_PATH}...")
    session = ort.InferenceSession(MODEL_PATH)
else:
    print(f"Warning: Model not found at path context {MODEL_PATH}. Checking working directory...")
    if os.path.exists(MODEL_FILENAME):
        session = ort.InferenceSession(MODEL_FILENAME)
    else:
        raise FileNotFoundError(
            f"Critical Deployment Error: '{MODEL_FILENAME}' was not found inside the root package context. "
            "Please ensure you upload your ONNX model to the same folder as app.py on Render."
        )

# Extract expected graph input structural key name automatically
input_name = session.get_inputs()[0].name

# Official 6-class structural disease mapping target matrix array
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

# --- RENDER KEEP-AWAKE SELF PING ALGORITHM ---
def keep_server_awake():
    """Background task to query local deployment route preventing spin down."""
    try:
        # Render provides RENDER_EXTERNAL_URL natively in environment setups
        self_url = os.environ.get('RENDER_EXTERNAL_URL')
        if self_url:
            ping_target = f"{self_url.rstrip('/')}/ping"
            print(f"[Keep-Awake] Pinging external routing gateway: {ping_target}")
            response = requests.get(ping_target, timeout=10)
            print(f"[Keep-Awake] Heartbeat acknowledged. Status Code: {response.status_code}")
        else:
            # Fallback local container route targeting default port configuration
            print("[Keep-Awake] RENDER_EXTERNAL_URL not found yet. Pinging local internal loop...")
            requests.get("http://127.0.0.1:10000/ping", timeout=5)
    except Exception as e:
        print(f"[Keep-Awake Warning] System loop heartbeat skip: {str(e)}")

# Initialize background scheduler task loop
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

        # 2. Structural Image Preprocessing
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))
        
        # Min-Max Normalization Scale Conversion to Float32: [0, 255] -> [0.0, 1.0]
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension vector tracking layer

        # 3. ONNX Inference Engine Parsing Pipeline Execution
        outputs = session.run(None, {input_name: img_array})
        predictions = outputs[0][0]  
        
        # Isolate index position showcasing maximum statistical strength
        max_idx = np.argmax(predictions)
        result_class = class_names[max_idx]
        confidence_score = float(predictions[max_idx]) * 100

        # 4. METHOD 1: DYNAMIC SEVERITY STAGE LOGIC
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

        # 5. FETCH AUTOMATED FARMER ADVICE
        if result_class == 'healthy':
            farmer_advice = ADVICE_DATABASE['healthy']['Safe']
        else:
            farmer_advice = ADVICE_DATABASE[result_class][lookup_stage]

        # Construct vector breakdown output for visual tracking bars
        breakdown = {}
        for idx, name in enumerate(class_names):
            breakdown[name] = round(float(predictions[idx]) * 100, 2)

        # 6. MOBILE APP READY UNIFIED JSON PAYLOAD OUTPUT
        return jsonify({
            'status': 'success',
            'prediction': result_class,
            'clean_prediction': result_class.replace('Tomato___', '').replace('_', ' '),
            'confidence': f"{confidence_score:.2f}%",
            'confidence_raw': round(confidence_score, 2),
            'severity_stage': severity_stage,
            'farmer_advice': farmer_advice,
            'breakdown': breakdown
        })

    except Exception as e:
        return jsonify({'error': f"Inference Failure Tracking: {str(e)}"}), 500

if __name__ == '__main__':
    # Render specifies alternative deployment port environments using env variables
    port = int(os.environ.get("PORT", 10000))
    print(f"AI Tomato Mobile Gateway processing actively inside host engine port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
