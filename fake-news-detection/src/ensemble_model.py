# Ensemble model implementation
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from config.config import PATHS
from src.models.traditional_models import TraditionalModels
from src.models.deep_learning_models import DeepLearningModels

class EnsembleModel:
    def __init__(self):
        self.traditional_models = TraditionalModels()
        self.dl_models = DeepLearningModels()
        self.ensemble_model = None
        self.weights = None
        
    def create_ensemble(self, X_train, y_train):
        """Create ensemble model from traditional ML models"""
        logging.info("Creating ensemble model...")
        
        # Load traditional models
        self.traditional_models.load_models()
        
        # Create voting classifier
        estimators = [
            ('rf', self.traditional_models.trained_models['random_forest']),
            ('gb', self.traditional_models.trained_models['gradient_boosting']),
            ('lr', self.traditional_models.trained_models['logistic_regression'])
        ]
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        # Train ensemble
        self.ensemble_model.fit(X_train, y_train)
        
        # Save ensemble model
        joblib.dump(self.ensemble_model, PATHS['ensemble_model'])
        
        logging.info("Ensemble model created and saved!")
        
    def weighted_prediction(self, text, tfidf_features):
        """Make weighted prediction using multiple models"""
        predictions = {}
        confidences = {}
        
        # Traditional ML prediction
        try:
            trad_pred = self.traditional_models.predict_single(tfidf_features, 'random_forest')
            predictions['traditional'] = trad_pred['prediction']
            confidences['traditional'] = trad_pred['confidence']
        except Exception as e:
            logging.warning(f"Traditional model prediction failed: {str(e)}")
            predictions['traditional'] = 0
            confidences['traditional'] = 0.5
        
        # Deep learning prediction
        try:
            dl_pred = self.dl_models.predict_single(text, 'hybrid')
            predictions['deep_learning'] = dl_pred['prediction']
            confidences['deep_learning'] = dl_pred['confidence']
        except Exception as e:
            logging.warning(f"Deep learning model prediction failed: {str(e)}")
            predictions['deep_learning'] = 0
            confidences['deep_learning'] = 0.5
        
        # Weighted average (you can adjust these weights based on model performance)
        weights = {
            'traditional': 0.4,
            'deep_learning': 0.6
        }
        
        # Calculate weighted prediction
        weighted_score = sum(
            predictions[model] * confidences[model] * weights[model]
            for model in predictions.keys()
        ) / sum(confidences[model] * weights[model] for model in predictions.keys())
        
        final_prediction = int(weighted_score > 0.5)
        final_confidence = max(weighted_score, 1 - weighted_score)
        
        return {
            'prediction': final_prediction,
            'confidence': float(final_confidence),
            'label': 'Real' if final_prediction == 1 else 'Fake',
            'individual_predictions': predictions,
            'individual_confidences': confidences
        }
    
    def load_ensemble(self):
        """Load trained ensemble model"""
        try:
            self.ensemble_model = joblib.load(PATHS['ensemble_model'])
            logging.info("Ensemble model loaded successfully!")
        except FileNotFoundError:
            logging.warning("Ensemble model not found. Create ensemble first.")
    
    def predict_ensemble(self, features):
        """Predict using ensemble model"""
        if self.ensemble_model is None:
            self.load_ensemble()
        
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not available")
        
        prediction = self.ensemble_model.predict(features)[0]
        confidence = self.ensemble_model.predict_proba(features)[0].max()
        
        return {
            'prediction': int(prediction),
            'confidence': float(confidence),
            'label': 'Real' if prediction == 1 else 'Fake'
        }
