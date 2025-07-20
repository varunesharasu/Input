# Fake News Detection Project

This project implements a comprehensive fake news detection system using various machine learning approaches including traditional models, deep learning, and BERT-based transformers.

## Project Structure

```
fake-news-detection/
├── config/
│   └── config.py                 # Configuration settings
├── src/
│   ├── data_preprocessing.py     # Data preprocessing pipeline
│   ├── models/
│   │   ├── traditional_models.py # Traditional ML models
│   │   ├── deep_learning_models.py # Deep learning models
│   │   └── bert_model.py         # BERT-based model
│   ├── ensemble_model.py         # Ensemble model implementation
│   ├── explanation.py            # Explanation and analysis
│   └── model_trainer.py          # Model training pipeline
├── app/
│   ├── streamlit_app.py          # Streamlit web application
│   ├── flask_app.py              # Flask API
│   └── templates/
│       └── index.html            # HTML template
├── scripts/
│   ├── setup_project.py          # Project setup script
│   ├── train_models.py           # Model training script
│   └── run_streamlit.py          # Streamlit runner
├── data/                         # Data directory
├── models/                       # Trained models
├── results/                      # Training results
├── logs/                         # Log files
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```
