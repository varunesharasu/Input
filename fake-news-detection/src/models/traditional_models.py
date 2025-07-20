# Traditional ML models
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import logging
from config.config import PATHS

class TraditionalModels:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(kernel='linear', probability=True, random_state=42),
            'naive_bayes': MultinomialNB()
        }
        self.trained_models = {}
        self.results = {}
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all traditional ML models"""
        logging.info("Training traditional ML models...")
        
        for name, model in self.models.items():
            logging.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Store trained model
            self.trained_models[name] = model
            
            logging.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Save all models
        self.save_models()
        
        return self.results
    
    def save_models(self):
        """Save all trained models"""
        for name, model in self.trained_models.items():
            model_path = PATHS['models_dir'] / f"{name}_model.pkl"
            joblib.dump(model, model_path)
        
        logging.info("All traditional models saved!")
    
    def load_models(self):
        """Load all trained models"""
        for name in self.models.keys():
            model_path = PATHS['models_dir'] / f"{name}_model.pkl"
            try:
                self.trained_models[name] = joblib.load(model_path)
            except FileNotFoundError:
                logging.warning(f"Model {name} not found at {model_path}")
    
    def predict_single(self, text_features, model_name='random_forest'):
        """Predict single text using specified model"""
        if model_name not in self.trained_models:
            self.load_models()
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.trained_models[model_name]
        prediction = model.predict(text_features)[0]
        
        if hasattr(model, 'predict_proba'):
            confidence = model.predict_proba(text_features)[0].max()
        else:
            confidence = 0.5
        
        return {
            'prediction': int(prediction),
            'confidence': float(confidence),
            'label': 'Real' if prediction == 1 else 'Fake'
        }
    
    def get_model_comparison(self):
        """Get comparison of all models"""
        if not self.results:
            logging.warning("No results available. Train models first.")
            return None
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df[['accuracy', 'precision', 'recall', 'f1_score']]
        comparison_df = comparison_df.round(4)
        
        return comparison_df
