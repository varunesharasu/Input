# Model training script
#!/usr/bin/env python3
"""
Script to train all models for fake news detection
"""

import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model_trainer import ModelTrainer

def main():
    """Main training function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting model training process...")
    
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Train all models
        results = trainer.train_all_models()
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        for model_type, models in results.items():
            print(f"\n{model_type.upper()} MODELS:")
            if model_type == 'bert':
                print(f"  BERT - F1: {models['f1_score']:.4f}, Accuracy: {models['accuracy']:.4f}")
            else:
                for model_name, metrics in models.items():
                    print(f"  {model_name.title()} - F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        # Get best model
        best_model, best_f1 = trainer.get_best_model()
        print(f"\nBest Model: {best_model[1]} ({best_model[0]}) with F1-Score: {best_f1:.4f}")
        
        print("\nModels saved successfully!")
        print("You can now run the web application:")
        print("  Streamlit: streamlit run app/streamlit_app.py")
        print("  Flask: python app/flask_app.py")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
