import os
import json
import joblib
from flask import Flask, render_template, request, jsonify

# Import preprocessing helper
from src.preprocessing import clean_text

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'model', 'vectorizer.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'model', 'metrics.json')

# Global variables for model, vectorizer, and metrics
model = None
vectorizer = None
metrics = {}

def load_artifacts():
    global model, vectorizer, metrics
    
    # If model files don't exist, trigger training pipeline
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        print("Model or vectorizer not found. Running training pipeline...")
        from src.train_model import train
        train()
        
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}
        
    print("Loaded model, vectorizer, and metrics successfully!")

# Load artifacts on startup
load_artifacts()

@app.route('/')
def home():
    """Home page: Textarea and prediction interface."""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page: Explaining the NLP + ML pipeline."""
    return render_template('about.html')

@app.route('/model')
def model_info():
    """Model page: Displays model parameters and evaluation metrics."""
    return render_template('model.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to receive review text and return prediction with confidence score."""
    data = request.get_json(silent=True) or request.form
    raw_text = data.get('text', '').strip()
    
    if not raw_text:
        return jsonify({
            'error': 'Please enter a valid review text before checking.'
        }), 400
        
    # Preprocess text
    cleaned = clean_text(raw_text)
    
    if not cleaned:
        return jsonify({
            'error': 'The review text contained no usable words after preprocessing.'
        }), 400
        
    # Transform using vectorizer
    tfidf_vector = vectorizer.transform([cleaned])
    
    # Predict probabilities and class
    # Class 0: Fake, Class 1: Genuine
    probabilities = model.predict_proba(tfidf_vector)[0]
    fake_prob = float(probabilities[0])
    genuine_prob = float(probabilities[1])
    
    # Class prediction
    if genuine_prob >= fake_prob:
        label = "Likely Genuine"
        is_genuine = True
        confidence = round(genuine_prob * 100, 1)
    else:
        label = "Likely Fake"
        is_genuine = False
        confidence = round(fake_prob * 100, 1)
        
    return jsonify({
        'raw_text': raw_text,
        'cleaned_text': cleaned,
        'label': label,
        'is_genuine': is_genuine,
        'confidence': confidence,
        'fake_probability': round(fake_prob * 100, 1),
        'genuine_probability': round(genuine_prob * 100, 1)
    })

if __name__ == '__main__':
    # Run Flask application locally on port 5001 to avoid conflicts
    print("Starting Fake Review Detection Flask server at http://127.0.0.1:5001")
    app.run(host='127.0.0.1', port=5001, debug=True)
