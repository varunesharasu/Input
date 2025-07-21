# # Configuration settings
# import os
# from pathlib import Path

# # Project Structure
# BASE_DIR = Path(__file__).parent.parent
# DATA_DIR = BASE_DIR / "data"
# MODELS_DIR = BASE_DIR / "models"
# LOGS_DIR = BASE_DIR / "logs"
# RESULTS_DIR = BASE_DIR / "results"

# # Create directories if they don't exist
# for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
#     dir_path.mkdir(exist_ok=True)

# # Model Configuration
# MODEL_CONFIG = {
#     'max_length': 512,
#     'batch_size': 32,
#     'epochs': 10,
#     'learning_rate': 2e-5,
#     'test_size': 0.2,
#     'random_state': 42
# }

# # Text Processing Configuration
# TEXT_CONFIG = {
#     'min_word_length': 2,
#     'max_features': 10000,
#     'ngram_range': (1, 2),
#     'stop_words': 'english'
# }

# # Web App Configuration
# WEB_CONFIG = {
#     'host': '0.0.0.0',
#     'port': 8501,
#     'debug': True
# }

# # File Paths
# PATHS = {
#     'true_data': DATA_DIR / "True.csv",
#     'false_data': DATA_DIR / "False.csv",
#     'processed_data': DATA_DIR / "processed_data.pkl",
#     'vectorizer': MODELS_DIR / "tfidf_vectorizer.pkl",
#     'bert_model': MODELS_DIR / "bert_model",
#     'lstm_model': MODELS_DIR / "lstm_model.h5",
#     'cnn_model': MODELS_DIR / "cnn_model.h5",
#     'ensemble_model': MODELS_DIR / "ensemble_model.pkl"
# }






import os
from pathlib import Path

# Project Structure
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model Configuration
MODEL_CONFIG = {
    'max_length': 512,
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 2e-5,
    'test_size': 0.2,
    'random_state': 42
}

# Text Processing Configuration
TEXT_CONFIG = {
    'min_word_length': 2,
    'max_features': 10000,
    'ngram_range': (1, 2),
    'stop_words': 'english'
}

# Web App Configuration
WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': 8501,
    'debug': True
}

# File Paths
PATHS = {
    'true_data': DATA_DIR / "True.csv",
    'false_data': DATA_DIR / "False.csv",
    'processed_data': DATA_DIR / "processed_data.pkl",
    'vectorizer': MODELS_DIR / "tfidf_vectorizer.pkl",
    'bert_model': MODELS_DIR / "bert_model",
    'lstm_model': MODELS_DIR / "lstm_model.h5",
    'cnn_model': MODELS_DIR / "cnn_model.h5",
    'ensemble_model': MODELS_DIR / "ensemble_model.pkl",
    'models_dir': MODELS_DIR,
    'results_dir': RESULTS_DIR
}
