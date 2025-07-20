# Model training pipeline
import logging
import pickle
from sklearn.model_selection import train_test_split
from src.data_preprocessing import DataPreprocessor
from src.models.traditional_models import TraditionalModels
from src.models.deep_learning_models import DeepLearningModels
from src.models.bert_model import BERTFakeNewsDetector
from src.ensemble_model import EnsembleModel
from config.config import PATHS, MODEL_CONFIG
import pandas as pd

class ModelTrainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.traditional_models = TraditionalModels()
        self.dl_models = DeepLearningModels()
        self.bert_model = BERTFakeNewsDetector()
        self.ensemble_model = EnsembleModel()
        self.results = {}
        
    def load_processed_data(self):
        """Load preprocessed data"""
        try:
            with open(PATHS['processed_data'], 'rb') as f:
                data = pickle.load(f)
            logging.info("Processed data loaded successfully!")
            return data
        except FileNotFoundError:
            logging.info("Processed data not found. Running preprocessing...")
            return self.preprocessor.run_preprocessing()
    
    def train_all_models(self):
        """Train all models"""
        logging.info("Starting comprehensive model training...")
        
        # Load processed data
        data = self.load_processed_data()
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        processed_df = data['processed_df']
        
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=MODEL_CONFIG['random_state']
        )
        
        # Train traditional ML models
        logging.info("Training traditional ML models...")
        traditional_results = self.traditional_models.train_all_models(
            X_train, y_train, X_test, y_test
        )
        self.results['traditional'] = traditional_results
        
        # Prepare text data for deep learning models
        train_texts = processed_df[processed_df.index.isin(range(len(y_train)))]['processed_content'].tolist()
        test_texts = processed_df[processed_df.index.isin(range(len(y_train), len(processed_df)))]['processed_content'].tolist()
        
        # Split train texts for validation
        train_texts_split, val_texts_split, _, _ = train_test_split(
            train_texts, y_train, test_size=0.2, random_state=MODEL_CONFIG['random_state']
        )
        
        # Train deep learning models
        logging.info("Training deep learning models...")
        dl_results = self.dl_models.train_all_models(
            train_texts_split, y_train_split,
            val_texts_split, y_val_split,
            test_texts, y_test
        )
        self.results['deep_learning'] = dl_results
        
        # Train BERT model (if computational resources allow)
        try:
            logging.info("Training BERT model...")
            self.bert_model.train(
                train_texts_split, y_train_split,
                val_texts_split, y_val_split
            )
            
            # Evaluate BERT model
            bert_predictions = []
            for text in test_texts:
                pred = self.bert_model.predict(text)
                bert_predictions.append(pred['prediction'])
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            bert_accuracy = accuracy_score(y_test, bert_predictions)
            bert_precision = precision_score(y_test, bert_predictions)
            bert_recall = recall_score(y_test, bert_predictions)
            bert_f1 = f1_score(y_test, bert_predictions)
            
            self.results['bert'] = {
                'accuracy': bert_accuracy,
                'precision': bert_precision,
                'recall': bert_recall,
                'f1_score': bert_f1
            }
            
        except Exception as e:
            logging.warning(f"BERT training failed: {str(e)}")
        
        # Create ensemble model
        logging.info("Creating ensemble model...")
        self.ensemble_model.create_ensemble(X_train, y_train)
        
        # Save results
        self.save_results()
        
        logging.info("All models trained successfully!")
        return self.results
    
    def save_results(self):
        """Save training results"""
        results_df = pd.DataFrame()
        
        for model_type, models in self.results.items():
            if model_type == 'bert':
                # BERT results are already in the right format
                temp_df = pd.DataFrame([models])
                temp_df.index = ['bert']
            else:
                temp_df = pd.DataFrame(models).T
            
            temp_df['model_type'] = model_type
            results_df = pd.concat([results_df, temp_df])
        
        results_df.to_csv(PATHS['results_dir'] / 'model_comparison.csv')
        
        with open(PATHS['results_dir'] / 'training_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        logging.info("Results saved successfully!")
    
    def get_best_model(self):
        """Get the best performing model"""
        if not self.results:
            logging.warning("No results available. Train models first.")
            return None
        
        best_f1 = 0
        best_model = None
        
        for model_type, models in self.results.items():
            if model_type == 'bert':
                if models['f1_score'] > best_f1:
                    best_f1 = models['f1_score']
                    best_model = ('bert', 'bert')
            else:
                for model_name, metrics in models.items():
                    if metrics['f1_score'] > best_f1:
                        best_f1 = metrics['f1_score']
                        best_model = (model_type, model_name)
        
        return best_model, best_f1

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = ModelTrainer()
    results = trainer.train_all_models()
    
    # Print results summary
    print("\n" + "="*50)
    print("TRAINING RESULTS SUMMARY")
    print("="*50)
    
    for model_type, models in results.items():
        print(f"\n{model_type.upper()} MODELS:")
        if model_type == 'bert':
            print(f"BERT - F1: {models['f1_score']:.4f}, Accuracy: {models['accuracy']:.4f}")
        else:
            for model_name, metrics in models.items():
                print(f"{model_name} - F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    
    best_model, best_f1 = trainer.get_best_model()
    print(f"\nBest Model: {best_model[1]} ({best_model[0]}) with F1-Score: {best_f1:.4f}")
