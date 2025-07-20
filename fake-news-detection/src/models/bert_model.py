# BERT-based model
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
from config.config import MODEL_CONFIG, PATHS

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTFakeNewsDetector:
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, texts, labels):
        """Prepare data for BERT training"""
        dataset = NewsDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=MODEL_CONFIG['max_length']
        )
        return dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_texts, train_labels, val_texts, val_labels):
        """Train BERT model"""
        logging.info("Initializing BERT model for training...")
        
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        
        # Prepare datasets
        train_dataset = self.prepare_data(train_texts, train_labels)
        val_dataset = self.prepare_data(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(PATHS['bert_model']),
            num_train_epochs=MODEL_CONFIG['epochs'],
            per_device_train_batch_size=MODEL_CONFIG['batch_size'],
            per_device_eval_batch_size=MODEL_CONFIG['batch_size'],
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(PATHS['bert_model'] / 'logs'),
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Train model
        logging.info("Starting BERT training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(str(PATHS['bert_model']))
        
        logging.info("BERT training completed!")
        return trainer
    
    def load_model(self):
        """Load trained BERT model"""
        try:
            self.model = BertForSequenceClassification.from_pretrained(str(PATHS['bert_model']))
            self.tokenizer = BertTokenizer.from_pretrained(str(PATHS['bert_model']))
            self.model.to(self.device)
            self.model.eval()
            logging.info("BERT model loaded successfully!")
        except Exception as e:
            logging.error(f"Error loading BERT model: {str(e)}")
            raise
    
    def predict(self, text):
        """Predict single text"""
        if self.model is None:
            self.load_model()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MODEL_CONFIG['max_length'],
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'label': 'Real' if predicted_class == 1 else 'Fake'
        }
    
    def predict_batch(self, texts):
        """Predict batch of texts"""
        if self.model is None:
            self.load_model()
        
        predictions = []
        for text in texts:
            pred = self.predict(text)
            predictions.append(pred)
        
        return predictions
