# Flask API
from flask import Flask, request, jsonify, render_template
import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import DataPreprocessor
from src.models.traditional_models import TraditionalModels
from src.models.deep_learning_models import DeepLearningModels
from src.models.bert_model import BERTFakeNewsDetector
from src.ensemble_model import EnsembleModel
from src.explanation import NewsExplainer
from config.config import PATHS
import pickle

app = Flask(__name__)

class FakeNewsAPI:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.traditional_models = TraditionalModels()
        self.dl_models = DeepLearningModels()
        self.bert_model = BERTFakeNewsDetector()
        self.ensemble_model = EnsembleModel()
        self.explainer = NewsExplainer()
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load traditional models
            self.traditional_models.load_models()
            
            # Load TF-IDF vectorizer
            with open(PATHS['vectorizer'], 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
    
    def preprocess_input_text(self, text):
        """Preprocess input text for prediction"""
        # Clean text
        cleaned_text = self.preprocessor.clean_text(text)
        processed_text = self.preprocessor.tokenize_and_lemmatize(cleaned_text)
        
        # Create TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([processed_text])
        
        return processed_text, tfidf_features
    
    def predict_with_all_models(self, text):
        """Make predictions with all available models"""
        processed_text, tfidf_features = self.preprocess_input_text(text)
        predictions = {}
        
        # Traditional ML models
        for model_name in ['random_forest', 'logistic_regression', 'gradient_boosting']:
            try:
                pred = self.traditional_models.predict_single(tfidf_features, model_name)
                predictions[model_name] = pred
            except Exception as e:
                print(f"Could not get prediction from {model_name}: {str(e)}")
        
        # Deep Learning models
        for model_name in ['lstm', 'cnn', 'hybrid']:
            try:
                pred = self.dl_models.predict_single(text, model_name)
                predictions[f"dl_{model_name}"] = pred
            except Exception as e:
                print(f"Could not get prediction from DL {model_name}: {str(e)}")
        
        # BERT model
        try:
            pred = self.bert_model.predict(text)
            predictions['bert'] = pred
        except Exception as e:
            print(f"Could not get BERT prediction: {str(e)}")
        
        # Ensemble prediction
        try:
            pred = self.ensemble_model.weighted_prediction(text, tfidf_features)
            predictions['ensemble'] = pred
        except Exception as e:
            print(f"Could not get ensemble prediction: {str(e)}")
        
        return predictions

# Initialize API
api = FakeNewsAPI()

@app.route('/')
def home():
    """Home page with simple interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Make predictions
        predictions = api.predict_with_all_models(text)
        
        if not predictions:
            return jsonify({'error': 'Could not make predictions'}), 500
        
        # Get main prediction (ensemble or first available)
        main_prediction = predictions.get('ensemble', list(predictions.values())[0])
        
        # Get explanation
        explanation = api.explainer.generate_explanation(text, main_prediction)
        
        # Prepare response
        response = {
            'main_prediction': main_prediction,
            'individual_predictions': predictions,
            'explanation': explanation,
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Fake News Detection API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
