# Fake News Detection System

A comprehensive fake news detection system based on the research paper "A systematic review of multimodal fake news detection on social media using deep learning models". This system implements multiple machine learning and deep learning approaches to detect fake news with high accuracy and provides detailed explanations for predictions.

## Features

- **Multiple Model Approaches**: Traditional ML, Deep Learning, and Transformer models
- **Comprehensive Analysis**: Text preprocessing, feature extraction, and pattern recognition
- **Explainable AI**: Detailed explanations for predictions with reasoning
- **Real-time Detection**: Instant analysis of news articles
- **Web Interface**: Both Streamlit and Flask applications
- **Model Comparison**: Performance metrics and visualization

## Project Structure

\`\`\`
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
\`\`\`

## Installation

1. **Clone or create the project structure**:
   \`\`\`bash
   mkdir fake-news-detection
   cd fake-news-detection
   \`\`\`

2. **Set up the project**:
   \`\`\`bash
   python scripts/setup_project.py
   \`\`\`

3. **Install dependencies**:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

## Data Preparation

1. **Place your datasets** in the `data/` directory:
   - `True.csv`: Real news articles
   - `False.csv`: Fake news articles

2. **Expected CSV format**:
   \`\`\`csv
   title,text,subject,date
   "News Title","News content text...","category","2023-01-01"
   \`\`\`

## Usage

### 1. Train Models

\`\`\`bash
# Run preprocessing and train all models
python scripts/train_models.py
\`\`\`

This will:
- Preprocess the data (cleaning, tokenization, feature extraction)
- Train traditional ML models (Random Forest, SVM, etc.)
- Train deep learning models (LSTM, CNN, Hybrid)
- Train BERT model (if resources allow)
- Create ensemble model
- Save all models and results

### 2. Run Web Application

#### Streamlit App (Recommended)
\`\`\`bash
streamlit run app/streamlit_app.py
\`\`\`
or
\`\`\`bash
python scripts/run_streamlit.py
\`\`\`

#### Flask API
\`\`\`bash
python app/flask_app.py
\`\`\`

### 3. Use the System

1. **Open the web interface** (usually at `http://localhost:8501` for Streamlit)
2. **Enter news text** in the text area
3. **Click "Analyze News"** to get predictions
4. **View results** including:
   - Main prediction (Fake/Real)
   - Confidence level
   - Individual model predictions
   - Detailed explanation
   - Text analysis and patterns

## Models Implemented

### Traditional Machine Learning
- **Logistic Regression**: Linear classification with regularization
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential ensemble method
- **Support Vector Machine**: Margin-based classification
- **Naive Bayes**: Probabilistic classifier

### Deep Learning
- **LSTM**: Long Short-Term Memory networks for sequential data
- **CNN**: Convolutional Neural Networks for text classification
- **Hybrid CNN-LSTM**: Combined approach for better performance

### Transformer Models
- **BERT**: Bidirectional Encoder Representations from Transformers
- **Fine-tuned BERT**: Specifically trained for fake news detection

### Ensemble Methods
- **Voting Classifier**: Combines multiple traditional models
- **Weighted Ensemble**: Combines all models with learned weights

## Features and Analysis

### Text Preprocessing
- Text cleaning and normalization
- Tokenization and lemmatization
- Stop word removal
- TF-IDF feature extraction

### Explanation System
- **Text Characteristics**: Word count, sentence structure, emotional language
- **Pattern Recognition**: Identifies fake news indicators
- **Feature Importance**: Shows which words/phrases influenced the decision
- **Confidence Analysis**: Explains prediction confidence
- **Visual Analysis**: Word clouds and pattern visualization

### Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- Model comparison and visualization
- Confusion matrices and ROC curves

## API Usage

### Flask API Endpoints

#### Predict News
\`\`\`bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article text here"}'
\`\`\`

#### Health Check
\`\`\`bash
curl http://localhost:5000/health
\`\`\`

### Response Format
\`\`\`json
{
  "main_prediction": {
    "prediction": 0,
    "confidence": 0.85,
    "label": "Fake"
  },
  "individual_predictions": {
    "random_forest": {"prediction": 0, "confidence": 0.82, "label": "Fake"},
    "bert": {"prediction": 0, "confidence": 0.91, "label": "Fake"}
  },
  "explanation": {
    "text_characteristics": {...},
    "reasoning": [...],
    "important_features": [...]
  },
  "status": "success"
}
\`\`\`

## Configuration

Edit `config/config.py` to customize:
- Model parameters (epochs, batch size, learning rate)
- Text processing settings
- File paths
- Web app settings

## Research Background

This system is based on the research paper:
**"A systematic review of multimodal fake news detection on social media using deep learning models"**

Key findings implemented:
- **Transformer models** (BERT) and **RNNs** are most effective
- **Multimodal approaches** improve accuracy
- **Ensemble methods** provide robust predictions
- **Feature extraction** is crucial for performance

## Performance

Expected performance on typical datasets:
- **Accuracy**: 85-95%
- **F1-Score**: 0.85-0.93
- **Precision**: 0.83-0.92
- **Recall**: 0.84-0.91

Performance varies based on:
- Dataset quality and size
- Text complexity and length
- Model configuration
- Available computational resources

## Troubleshooting

### Common Issues

1. **Models not loading**:
   - Ensure models are trained first: `python scripts/train_models.py`
   - Check file paths in `config/config.py`

2. **Memory errors during training**:
   - Reduce batch size in `config/config.py`
   - Use smaller models or reduce max_length

3. **BERT training fails**:
   - Requires significant GPU memory
   - Consider using smaller BERT variants
   - Skip BERT training if resources are limited

4. **Web app not starting**:
   - Check port availability
   - Install all requirements
   - Ensure models are trained

### Performance Optimization

1. **For faster training**:
   - Use GPU if available
   - Reduce dataset size for testing
   - Use pre-trained embeddings

2. **For better accuracy**:
   - Increase training epochs
   - Use larger datasets
   - Fine-tune hyperparameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@article{nasser2025systematic,
  title={A systematic review of multimodal fake news detection on social media using deep learning models},
  author={Nasser, Maged and Arshad, Noreen Izza and Ali, Abdulalem and others},
  journal={Results in Engineering},
  volume={26},
  pages={104752},
  year={2025},
  publisher={Elsevier}
}
