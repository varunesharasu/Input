# Streamlit web application
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import DataPreprocessor
from src.models.traditional_models import TraditionalModels
from src.models.deep_learning_models import DeepLearningModels
from src.models.bert_model import BERTFakeNewsDetector
from src.ensemble_model import EnsembleModel
from src.explanation import NewsExplainer
from config.config import PATHS

# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .fake-news {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .real-news {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .confidence-bar {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FakeNewsApp:
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
            
            st.success("Models loaded successfully!")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
    
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
                st.warning(f"Could not get prediction from {model_name}: {str(e)}")
        
        # Deep Learning models
        for model_name in ['lstm', 'cnn', 'hybrid']:
            try:
                pred = self.dl_models.predict_single(text, model_name)
                predictions[f"dl_{model_name}"] = pred
            except Exception as e:
                st.warning(f"Could not get prediction from DL {model_name}: {str(e)}")
        
        # BERT model
        try:
            pred = self.bert_model.predict(text)
            predictions['bert'] = pred
        except Exception as e:
            st.warning(f"Could not get BERT prediction: {str(e)}")
        
        # Ensemble prediction
        try:
            pred = self.ensemble_model.weighted_prediction(text, tfidf_features)
            predictions['ensemble'] = pred
        except Exception as e:
            st.warning(f"Could not get ensemble prediction: {str(e)}")
        
        return predictions
    
    def display_prediction_results(self, predictions):
        """Display prediction results"""
        if not predictions:
            st.error("No predictions available")
            return
        
        # Get ensemble or best prediction
        main_prediction = predictions.get('ensemble', list(predictions.values())[0])
        
        # Display main prediction
        prediction_class = "fake-news" if main_prediction['label'] == 'Fake' else "real-news"
        
        st.markdown(f"""
        <div class="prediction-box {prediction_class}">
            Prediction: {main_prediction['label']} News
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence bar
        confidence = main_prediction['confidence']
        st.markdown("### Confidence Level")
        st.progress(confidence)
        st.write(f"Confidence: {confidence:.2%}")
        
        # Individual model predictions
        st.markdown("### Individual Model Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Traditional ML Models")
            for model_name, pred in predictions.items():
                if not model_name.startswith('dl_') and model_name not in ['bert', 'ensemble']:
                    label_color = "🔴" if pred['label'] == 'Fake' else "🟢"
                    st.write(f"{label_color} **{model_name.title()}**: {pred['label']} ({pred['confidence']:.2%})")
        
        with col2:
            st.markdown("#### Deep Learning Models")
            for model_name, pred in predictions.items():
                if model_name.startswith('dl_') or model_name in ['bert']:
                    label_color = "🔴" if pred['label'] == 'Fake' else "🟢"
                    display_name = model_name.replace('dl_', '').title()
                    st.write(f"{label_color} **{display_name}**: {pred['label']} ({pred['confidence']:.2%})")
        
        # Visualization
        self.create_prediction_visualization(predictions)
    
    def create_prediction_visualization(self, predictions):
        """Create visualization of predictions"""
        # Prepare data for visualization
        model_names = []
        confidences = []
        labels = []
        
        for model_name, pred in predictions.items():
            if model_name != 'ensemble':
                model_names.append(model_name.replace('dl_', '').title())
                confidences.append(pred['confidence'])
                labels.append(pred['label'])
        
        # Create bar chart
        fig = px.bar(
            x=model_names,
            y=confidences,
            color=labels,
            title="Model Predictions Comparison",
            labels={'x': 'Models', 'y': 'Confidence'},
            color_discrete_map={'Fake': '#ff4444', 'Real': '#44ff44'}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_explanation(self, text, main_prediction):
        """Display explanation for the prediction"""
        st.markdown("### Why This Prediction?")
        
        # Get explanation
        explanation = self.explainer.generate_explanation(text, main_prediction)
        
        # Text characteristics
        st.markdown("#### Text Analysis")
        characteristics = explanation['text_characteristics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Word Count", characteristics['word_count'])
        with col2:
            st.metric("Sentences", characteristics['sentence_count'])
        with col3:
            st.metric("Emotional Words", characteristics['emotional_words_count'])
        with col4:
            st.metric("Avg Word Length", f"{characteristics['avg_word_length']:.1f}")
        
        # Reasoning
        if explanation['reasoning']:
            st.markdown("#### Key Indicators")
            for reason in explanation['reasoning']:
                st.write(f"• {reason}")
        
        # Pattern analysis
        patterns = self.explainer.get_similar_patterns(text, main_prediction['label'])
        
        if patterns['fake_patterns'] or patterns['real_patterns']:
            st.markdown("#### Detected Patterns")
            
            if patterns['fake_patterns']:
                st.markdown("**Fake News Indicators Found:**")
                for pattern in patterns['fake_patterns']:
                    st.write(f"🔴 '{pattern}'")
            
            if patterns['real_patterns']:
                st.markdown("**Real News Indicators Found:**")
                for pattern in patterns['real_patterns']:
                    st.write(f"🟢 '{pattern}'")
        
        # Word cloud
        st.markdown("#### Word Cloud")
        wordcloud = self.explainer.create_word_cloud(text, main_prediction['label'])
        
        if wordcloud:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    def display_model_performance(self):
        """Display model performance metrics"""
        try:
            # Load results
            with open(PATHS['results_dir'] / 'training_results.pkl', 'rb') as f:
                results = pickle.load(f)
            
            st.markdown("### Model Performance Comparison")
            
            # Create performance dataframe
            performance_data = []
            
            for model_type, models in results.items():
                if model_type == 'bert':
                    performance_data.append({
                        'Model': 'BERT',
                        'Type': 'Deep Learning',
                        'Accuracy': models['accuracy'],
                        'Precision': models['precision'],
                        'Recall': models['recall'],
                        'F1-Score': models['f1_score']
                    })
                else:
                    for model_name, metrics in models.items():
                        performance_data.append({
                            'Model': model_name.title(),
                            'Type': 'Traditional ML' if model_type == 'traditional' else 'Deep Learning',
                            'Accuracy': metrics['accuracy'],
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall'],
                            'F1-Score': metrics['f1_score']
                        })
            
            df = pd.DataFrame(performance_data)
            
            # Display table
            st.dataframe(df.round(4))
            
            # Create performance visualization
            fig = px.bar(
                df,
                x='Model',
                y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                title="Model Performance Metrics",
                barmode='group'
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except FileNotFoundError:
            st.warning("Performance results not found. Train models first.")

def main():
    # Initialize app
    app = FakeNewsApp()
    
    # Header
    st.markdown('<h1 class="main-header">📰 Fake News Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Performance", "About"])
    
    if page == "Prediction":
        st.markdown("### Enter News Text for Analysis")
        
        # Text input
        input_text = st.text_area(
            "Paste your news article here:",
            height=200,
            placeholder="Enter the news article text you want to analyze..."
        )
        
        # Analysis button
        if st.button("Analyze News", type="primary"):
            if input_text.strip():
                with st.spinner("Analyzing news article..."):
                    # Make predictions
                    predictions = app.predict_with_all_models(input_text)
                    
                    if predictions:
                        # Display results
                        app.display_prediction_results(predictions)
                        
                        # Get main prediction for explanation
                        main_prediction = predictions.get('ensemble', list(predictions.values())[0])
                        
                        # Display explanation
                        with st.expander("Detailed Analysis", expanded=True):
                            app.display_explanation(input_text, main_prediction)
                    else:
                        st.error("Could not analyze the text. Please check if models are properly loaded.")
            else:
                st.warning("Please enter some text to analyze.")
        
        # Example texts
        st.markdown("### Try These Examples")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Example: Fake News"):
                example_fake = """
                SHOCKING: Scientists discover that drinking water can be deadly! 
                You won't believe what they found in your tap water. This is something 
                they don't want you to know! Share this immediately to save lives!
                """
                st.text_area("Example text:", value=example_fake, height=100, key="fake_example")
        
        with col2:
            if st.button("Example: Real News"):
                example_real = """
                According to a new study published in the Journal of Environmental Health, 
                researchers have found that regular water quality testing is essential for 
                public safety. The study, conducted over two years, analyzed water samples 
                from 500 municipalities and found that 95% met safety standards.
                """
                st.text_area("Example text:", value=example_real, height=100, key="real_example")
    
    elif page == "Model Performance":
        app.display_model_performance()
    
    elif page == "About":
        st.markdown("""
        ### About This System
        
        This Fake News Detection System is based on the research paper:
        **"A systematic review of multimodal fake news detection on social media using deep learning models"**
        
        #### Features:
        - **Multiple Model Approaches**: Traditional ML, Deep Learning, and Transformer models
        - **Comprehensive Analysis**: Text preprocessing, feature extraction, and pattern recognition
        - **Explainable AI**: Detailed explanations for predictions
        - **Real-time Detection**: Instant analysis of news articles
        
        #### Models Used:
        - **Traditional ML**: Random Forest, Logistic Regression, Gradient Boosting, SVM, Naive Bayes
        - **Deep Learning**: LSTM, CNN, Hybrid CNN-LSTM
        - **Transformers**: BERT-based models
        - **Ensemble**: Weighted combination of multiple models
        
        #### How It Works:
        1. **Text Preprocessing**: Cleaning, tokenization, and lemmatization
        2. **Feature Extraction**: TF-IDF, word embeddings, and BERT embeddings
        3. **Model Prediction**: Multiple models analyze the text
        4. **Ensemble Decision**: Weighted combination of predictions
        5. **Explanation**: Analysis of why the prediction was made
        
        #### Evaluation Metrics:
        - Accuracy, Precision, Recall, F1-Score
        - Based on True.csv and False.csv datasets
        
        #### Technology Stack:
        - **Backend**: Python, TensorFlow, PyTorch, Scikit-learn
        - **NLP**: NLTK, spaCy, Transformers
        - **Frontend**: Streamlit
        - **Visualization**: Plotly, Matplotlib, WordCloud
        """)

if __name__ == "__main__":
    main()
