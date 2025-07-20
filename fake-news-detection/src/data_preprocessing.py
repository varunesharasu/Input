# Data preprocessing pipeline
import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import logging
from config.config import MODEL_CONFIG, TEXT_CONFIG, PATHS

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = None
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) >= TEXT_CONFIG['min_word_length']
        ]
        
        return ' '.join(tokens)
    
    def load_and_combine_data(self):
        """Load and combine True.csv and False.csv files"""
        try:
            # Load datasets
            true_df = pd.read_csv(PATHS['true_data'])
            false_df = pd.read_csv(PATHS['false_data'])
            
            # Add labels
            true_df['label'] = 1  # Real news
            false_df['label'] = 0  # Fake news
            
            # Combine datasets
            combined_df = pd.concat([true_df, false_df], ignore_index=True)
            
            # Combine title and text for better feature extraction
            combined_df['content'] = combined_df['title'].fillna('') + ' ' + combined_df['text'].fillna('')
            
            logging.info(f"Loaded {len(true_df)} real news and {len(false_df)} fake news articles")
            return combined_df
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, df):
        """Complete preprocessing pipeline"""
        logging.info("Starting data preprocessing...")
        
        # Clean text
        df['cleaned_content'] = df['content'].apply(self.clean_text)
        
        # Tokenize and lemmatize
        df['processed_content'] = df['cleaned_content'].apply(self.tokenize_and_lemmatize)
        
        # Remove empty content
        df = df[df['processed_content'].str.len() > 0]
        
        logging.info(f"Preprocessing completed. Final dataset size: {len(df)}")
        return df
    
    def create_features(self, df):
        """Create TF-IDF features"""
        logging.info("Creating TF-IDF features...")
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=TEXT_CONFIG['max_features'],
            ngram_range=TEXT_CONFIG['ngram_range'],
            stop_words=TEXT_CONFIG['stop_words']
        )
        
        # Fit and transform
        tfidf_features = self.tfidf_vectorizer.fit_transform(df['processed_content'])
        
        # Save vectorizer
        with open(PATHS['vectorizer'], 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        return tfidf_features
    
    def split_data(self, features, labels):
        """Split data into train and test sets"""
        return train_test_split(
            features, labels,
            test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state'],
            stratify=labels
        )
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        # Load data
        df = self.load_and_combine_data()
        
        # Preprocess text
        df = self.preprocess_data(df)
        
        # Create features
        features = self.create_features(df)
        labels = df['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(features, labels)
        
        # Save processed data
        processed_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.tfidf_vectorizer.get_feature_names_out(),
            'processed_df': df
        }
        
        with open(PATHS['processed_data'], 'wb') as f:
            pickle.dump(processed_data, f)
        
        logging.info("Data preprocessing completed and saved!")
        return processed_data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocessor = DataPreprocessor()
    preprocessor.run_preprocessing()
