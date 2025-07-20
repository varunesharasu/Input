# Explanation and analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from config.config import PATHS

class NewsExplainer:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.load_vectorizer()
        
    def load_vectorizer(self):
        """Load TF-IDF vectorizer"""
        try:
            with open(PATHS['vectorizer'], 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
        except FileNotFoundError:
            logging.warning("TF-IDF vectorizer not found")
    
    def get_important_features(self, text, model, top_n=10):
        """Get most important features for prediction"""
        if self.tfidf_vectorizer is None:
            return []
        
        # Transform text
        text_features = self.tfidf_vectorizer.transform([text])
        
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Get feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Get non-zero features
            non_zero_indices = text_features.nonzero()[1]
            feature_importance_pairs = [
                (feature_names[i], importances[i] * text_features[0, i])
                for i in non_zero_indices
            ]
            
            # Sort by importance
            feature_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return feature_importance_pairs[:top_n]
        
        # For other models, use TF-IDF scores
        else:
            non_zero_indices = text_features.nonzero()[1]
            tfidf_scores = [
                (feature_names[i], text_features[0, i])
                for i in non_zero_indices
            ]
            
            # Sort by TF-IDF score
            tfidf_scores.sort(key=lambda x: x[1], reverse=True)
            
            return tfidf_scores[:top_n]
    
    def analyze_text_characteristics(self, text):
        """Analyze text characteristics that might indicate fake news"""
        characteristics = {}
        
        # Basic statistics
        characteristics['word_count'] = len(text.split())
        characteristics['char_count'] = len(text)
        characteristics['sentence_count'] = len(re.split(r'[.!?]+', text))
        characteristics['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # Punctuation analysis
        exclamation_count = text.count('!')
        question_count = text.count('?')
        characteristics['exclamation_ratio'] = exclamation_count / len(text) if len(text) > 0 else 0
        characteristics['question_ratio'] = question_count / len(text) if len(text) > 0 else 0
        
        # Capitalization analysis
        caps_count = sum(1 for c in text if c.isupper())
        characteristics['caps_ratio'] = caps_count / len(text) if len(text) > 0 else 0
        
        # Emotional words (simple approach)
        emotional_words = [
            'amazing', 'incredible', 'shocking', 'unbelievable', 'devastating',
            'outrageous', 'fantastic', 'terrible', 'horrible', 'wonderful'
        ]
        emotional_count = sum(1 for word in emotional_words if word.lower() in text.lower())
        characteristics['emotional_words_count'] = emotional_count
        
        return characteristics
    
    def generate_explanation(self, text, prediction_result, model=None):
        """Generate comprehensive explanation for prediction"""
        explanation = {
            'prediction': prediction_result['label'],
            'confidence': prediction_result['confidence'],
            'text_characteristics': self.analyze_text_characteristics(text),
            'important_features': [],
            'reasoning': []
        }
        
        # Get important features if model is provided
        if model is not None:
            try:
                explanation['important_features'] = self.get_important_features(text, model)
            except Exception as e:
                logging.warning(f"Could not extract important features: {str(e)}")
        
        # Generate reasoning based on characteristics
        characteristics = explanation['text_characteristics']
        reasoning = []
        
        # Word count analysis
        if characteristics['word_count'] < 50:
            reasoning.append("Very short text - may lack sufficient context for reliable news")
        elif characteristics['word_count'] > 1000:
            reasoning.append("Very long text - typical of detailed news articles")
        
        # Emotional language analysis
        if characteristics['emotional_words_count'] > 3:
            reasoning.append("High use of emotional language - common in sensationalized content")
        
        # Punctuation analysis
        if characteristics['exclamation_ratio'] > 0.01:
            reasoning.append("High exclamation mark usage - may indicate sensationalized content")
        
        if characteristics['caps_ratio'] > 0.05:
            reasoning.append("High capitalization ratio - may indicate emphasis or shouting")
        
        # Confidence-based reasoning
        if prediction_result['confidence'] > 0.8:
            reasoning.append("High confidence prediction based on strong textual indicators")
        elif prediction_result['confidence'] < 0.6:
            reasoning.append("Low confidence prediction - text may have mixed indicators")
        
        explanation['reasoning'] = reasoning
        
        return explanation
    
    def create_word_cloud(self, text, prediction_label):
        """Create word cloud for visualization"""
        try:
            # Clean text for word cloud
            cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis' if prediction_label == 'Real' else 'Reds',
                max_words=100
            ).generate(cleaned_text)
            
            return wordcloud
        except Exception as e:
            logging.error(f"Error creating word cloud: {str(e)}")
            return None
    
    def get_similar_patterns(self, text, prediction_label):
        """Identify patterns similar to known fake/real news"""
        patterns = {
            'fake_indicators': [
                'you won\'t believe',
                'shocking truth',
                'they don\'t want you to know',
                'breaking:',
                'urgent:',
                'must read',
                'click here',
                'share if you agree'
            ],
            'real_indicators': [
                'according to',
                'research shows',
                'study finds',
                'data indicates',
                'experts say',
                'official statement',
                'confirmed by',
                'reported by'
            ]
        }
        
        text_lower = text.lower()
        found_patterns = {
            'fake_patterns': [pattern for pattern in patterns['fake_indicators'] if pattern in text_lower],
            'real_patterns': [pattern for pattern in patterns['real_indicators'] if pattern in text_lower]
        }
        
        return found_patterns
