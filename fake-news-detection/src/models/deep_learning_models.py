# # Deep learning models
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Input, Concatenate
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import numpy as np
# import pandas as pd
# import pickle
# import logging
# from config.config import MODEL_CONFIG, PATHS

# class DeepLearningModels:
#     def __init__(self, max_words=10000, max_length=500):
#         self.max_words = max_words
#         self.max_length = max_length
#         self.tokenizer = None
#         self.models = {}
        
#     def prepare_text_data(self, texts):
#         """Prepare text data for deep learning models"""
#         if self.tokenizer is None:
#             self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
#             self.tokenizer.fit_on_texts(texts)
        
#         sequences = self.tokenizer.texts_to_sequences(texts)
#         padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
#         return padded_sequences
    
#     def build_lstm_model(self, embedding_dim=100):
#         """Build LSTM model for fake news detection"""
#         model = Sequential([
#             Embedding(self.max_words, embedding_dim, input_length=self.max_length),
#             LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
#             LSTM(64, dropout=0.2, recurrent_dropout=0.2),
#             Dense(32, activation='relu'),
#             Dropout(0.5),
#             Dense(1, activation='sigmoid')
#         ])
        
#         model.compile(
#             optimizer='adam',
#             loss='binary_crossentropy',
#             metrics=['accuracy']
#         )
        
#         return model
    
#     def build_cnn_model(self, embedding_dim=100):
#         """Build CNN model for fake news detection"""
#         model = Sequential([
#             Embedding(self.max_words, embedding_dim, input_length=self.max_length),
#             Conv1D(128, 5, activation='relu'),
#             MaxPooling1D(5),
#             Conv1D(128, 5, activation='relu'),
#             MaxPooling1D(5),
#             Conv1D(128, 5, activation='relu'),
#             GlobalMaxPooling1D(),
#             Dense(128, activation='relu'),
#             Dropout(0.5),
#             Dense(1, activation='sigmoid')
#         ])
        
#         model.compile(
#             optimizer='adam',
#             loss='binary_crossentropy',
#             metrics=['accuracy']
#         )
        
#         return model
    
#     def build_hybrid_model(self, embedding_dim=100):
#         """Build hybrid CNN-LSTM model"""
#         # Input layer
#         input_layer = Input(shape=(self.max_length,))
        
#         # Embedding layer
#         embedding = Embedding(self.max_words, embedding_dim)(input_layer)
        
#         # CNN branch
#         cnn_branch = Conv1D(64, 3, activation='relu')(embedding)
#         cnn_branch = MaxPooling1D(2)(cnn_branch)
#         cnn_branch = Conv1D(64, 3, activation='relu')(cnn_branch)
#         cnn_branch = GlobalMaxPooling1D()(cnn_branch)
        
#         # LSTM branch
#         lstm_branch = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(embedding)
        
#         # Concatenate branches
#         merged = Concatenate()([cnn_branch, lstm_branch])
        
#         # Dense layers
#         dense = Dense(64, activation='relu')(merged)
#         dense = Dropout(0.5)(dense)
#         output = Dense(1, activation='sigmoid')(dense)
        
#         model = Model(inputs=input_layer, outputs=output)
#         model.compile(
#             optimizer='adam',
#             loss='binary_crossentropy',
#             metrics=['accuracy']
#         )
        
#         return model
    
#     def train_model(self, model, X_train, y_train, X_val, y_val, model_name):
#         """Train a deep learning model"""
#         logging.info(f"Training {model_name} model...")
        
#         # Callbacks
#         early_stopping = EarlyStopping(
#             monitor='val_loss',
#             patience=3,
#             restore_best_weights=True
#         )
        
#         model_checkpoint = ModelCheckpoint(
#             str(PATHS['models_dir'] / f"{model_name}_model.h5"),
#             monitor='val_accuracy',
#             save_best_only=True,
#             verbose=1
#         )
        
#         # Train model
#         history = model.fit(
#             X_train, y_train,
#             batch_size=MODEL_CONFIG['batch_size'],
#             epochs=MODEL_CONFIG['epochs'],
#             validation_data=(X_val, y_val),
#             callbacks=[early_stopping, model_checkpoint],
#             verbose=1
#         )
        
#         # Store model
#         self.models[model_name] = model
        
#         logging.info(f"{model_name} training completed!")
#         return history
    
#     def evaluate_model(self, model, X_test, y_test):
#         """Evaluate model performance"""
#         # Make predictions
#         y_pred_proba = model.predict(X_test)
#         y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
#         # Calculate metrics
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
        
#         return {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1,
#             'predictions': y_pred,
#             'probabilities': y_pred_proba.flatten()
#         }
    
#     def train_all_models(self, train_texts, y_train, val_texts, y_val, test_texts, y_test):
#         """Train all deep learning models"""
#         # Prepare data
#         X_train = self.prepare_text_data(train_texts)
#         X_val = self.prepare_text_data(val_texts)
#         X_test = self.prepare_text_data(test_texts)
        
#         # Save tokenizer
#         with open(PATHS['models_dir'] / 'tokenizer.pkl', 'wb') as f:
#             pickle.dump(self.tokenizer, f)
        
#         results = {}
        
#         # Train LSTM model
#         lstm_model = self.build_lstm_model()
#         self.train_model(lstm_model, X_train, y_train, X_val, y_val, 'lstm')
#         results['lstm'] = self.evaluate_model(lstm_model, X_test, y_test)
        
#         # Train CNN model
#         cnn_model = self.build_cnn_model()
#         self.train_model(cnn_model, X_train, y_train, X_val, y_val, 'cnn')
#         results['cnn'] = self.evaluate_model(cnn_model, X_test, y_test)
        
#         # Train Hybrid model
#         hybrid_model = self.build_hybrid_model()
#         self.train_model(hybrid_model, X_train, y_train, X_val, y_val, 'hybrid')
#         results['hybrid'] = self.evaluate_model(hybrid_model, X_test, y_test)
        
#         return results
    
#     def load_model(self, model_name):
#         """Load a trained model"""
#         try:
#             model_path = PATHS['models_dir'] / f"{model_name}_model.h5"
#             model = tf.keras.models.load_model(str(model_path))
            
#             # Load tokenizer
#             with open(PATHS['models_dir'] / 'tokenizer.pkl', 'rb') as f:
#                 self.tokenizer = pickle.load(f)
            
#             self.models[model_name] = model
#             logging.info(f"{model_name} model loaded successfully!")
#             return model
#         except Exception as e:
#             logging.error(f"Error loading {model_name} model: {str(e)}")
#             raise
    
#     def predict_single(self, text, model_name='hybrid'):
#         """Predict single text using specified model"""
#         if model_name not in self.models:
#             self.load_model(model_name)
        
#         # Prepare text
#         sequence = self.tokenizer.texts_to_sequences([text])
#         padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        
#         # Make prediction
#         model = self.models[model_name]
#         prediction_proba = model.predict(padded_sequence)[0][0]
#         prediction = int(prediction_proba > 0.5)
        
#         return {
#             'prediction': prediction,
#             'confidence': float(max(prediction_proba, 1 - prediction_proba)),
#             'label': 'Real' if prediction == 1 else 'Fake'
#         }











import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Input, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import pickle
import logging
from config.config import MODEL_CONFIG, PATHS, MODELS_DIR

class DeepLearningModels:
    def __init__(self, max_words=10000, max_length=500):
        self.max_words = max_words
        self.max_length = max_length
        self.tokenizer = None
        self.models = {}
        
    def prepare_text_data(self, texts):
        """Prepare text data for deep learning models"""
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        return padded_sequences
    
    def build_lstm_model(self, embedding_dim=100):
        """Build LSTM model for fake news detection"""
        model = Sequential([
            Embedding(self.max_words, embedding_dim, input_length=self.max_length),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn_model(self, embedding_dim=100):
        """Build CNN model for fake news detection"""
        model = Sequential([
            Embedding(self.max_words, embedding_dim, input_length=self.max_length),
            Conv1D(128, 5, activation='relu'),
            MaxPooling1D(5),
            Conv1D(128, 5, activation='relu'),
            MaxPooling1D(5),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_hybrid_model(self, embedding_dim=100):
        """Build hybrid CNN-LSTM model"""
        # Input layer
        input_layer = Input(shape=(self.max_length,))
        
        # Embedding layer
        embedding = Embedding(self.max_words, embedding_dim)(input_layer)
        
        # CNN branch
        cnn_branch = Conv1D(64, 3, activation='relu')(embedding)
        cnn_branch = MaxPooling1D(2)(cnn_branch)
        cnn_branch = Conv1D(64, 3, activation='relu')(cnn_branch)
        cnn_branch = GlobalMaxPooling1D()(cnn_branch)
        
        # LSTM branch
        lstm_branch = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(embedding)
        
        # Concatenate branches
        merged = Concatenate()([cnn_branch, lstm_branch])
        
        # Dense layers
        dense = Dense(64, activation='relu')(merged)
        dense = Dropout(0.5)(dense)
        output = Dense(1, activation='sigmoid')(dense)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, X_train, y_train, X_val, y_val, model_name):
        """Train a deep learning model"""
        logging.info(f"Training {model_name} model...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            str(PATHS['models_dir'] / f"{model_name}_model.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=MODEL_CONFIG['batch_size'],
            epochs=MODEL_CONFIG['epochs'],
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Store model
        self.models[model_name] = model
        
        logging.info(f"{model_name} training completed!")
        return history
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        # Make predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba.flatten()
        }
    
    def train_all_models(self, train_texts, y_train, val_texts, y_val, test_texts, y_test):
        """Train all deep learning models"""
        # Prepare data
        X_train = self.prepare_text_data(train_texts)
        X_val = self.prepare_text_data(val_texts)
        X_test = self.prepare_text_data(test_texts)
        
        # Save tokenizer
        with open(MODELS_DIR / 'tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        results = {}
        
        # Train LSTM model
        lstm_model = self.build_lstm_model()
        self.train_model(lstm_model, X_train, y_train, X_val, y_val, 'lstm')
        results['lstm'] = self.evaluate_model(lstm_model, X_test, y_test)
        
        # Train CNN model
        cnn_model = self.build_cnn_model()
        self.train_model(cnn_model, X_train, y_train, X_val, y_val, 'cnn')
        results['cnn'] = self.evaluate_model(cnn_model, X_test, y_test)
        
        # Train Hybrid model
        hybrid_model = self.build_hybrid_model()
        self.train_model(hybrid_model, X_train, y_train, X_val, y_val, 'hybrid')
        results['hybrid'] = self.evaluate_model(hybrid_model, X_test, y_test)
        
        return results
    
    def load_model(self, model_name):
        """Load a trained model"""
        try:
            model_path = MODELS_DIR / f"{model_name}_model.h5"
            model = tf.keras.models.load_model(str(model_path))
            
            # Load tokenizer
            with open(MODELS_DIR / 'tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            self.models[model_name] = model
            logging.info(f"{model_name} model loaded successfully!")
            return model
        except Exception as e:
            logging.error(f"Error loading {model_name} model: {str(e)}")
            raise
    
    def predict_single(self, text, model_name='hybrid'):
        """Predict single text using specified model"""
        if model_name not in self.models:
            self.load_model(model_name)
        
        # Prepare text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        
        # Make prediction
        model = self.models[model_name]
        prediction_proba = model.predict(padded_sequence)[0][0]
        prediction = int(prediction_proba > 0.5)
        
        return {
            'prediction': prediction,
            'confidence': float(max(prediction_proba, 1 - prediction_proba)),
            'label': 'Real' if prediction == 1 else 'Fake'
        }
