import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path
from .config import MODELS_DIR, MODEL_CONFIG

class BETOMultiLabelClassifier:
    def __init__(self, num_labels):
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        self.is_trained = False
        
        # Cargar modelo y tokenizer
        self.load_pretrained()
    
    def load_pretrained(self):
        """Carga el modelo BETO pre-entrenado"""
        print("🔄 Cargando modelo BETO...")
        
        model_name = MODEL_CONFIG['berto']['model_name']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        )
        
        print("✅ BETO cargado correctamente")
    
    def prepare_dataset(self, texts, labels=None, max_length=256):
        """Prepara el dataset para BETO"""
        encodings = self.tokenizer(
            texts.tolist() if hasattr(texts, 'tolist') else texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        if labels is not None:
            encodings['labels'] = torch.tensor(labels, dtype=torch.float)
        
        return encodings
    
    def compute_metrics(self, eval_pred):
        """Calcula métricas para evaluación"""
        predictions, labels = eval_pred
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        
        # Convertir probabilidades a etiquetas binarias
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= 0.5)] = 1
        
        # Calcular métricas
        f1 = f1_score(labels, y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(labels, y_pred)
        
        return {
            'f1_macro': f1,
            'accuracy': accuracy,
        }
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None):
        """Entrena el modelo BETO"""
        print("⏳ Entrenando modelo BETO...")
        
        # Preparar datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        
        if val_texts is not None and val_labels is not None:
            val_dataset = self.prepare_dataset(val_texts, val_labels)
        else:
            val_dataset = None
        
        # Configurar argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir=MODELS_DIR / "berto_checkpoints",
            num_train_epochs=MODEL_CONFIG['berto']['epochs'],
            per_device_train_batch_size=MODEL_CONFIG['berto']['batch_size'],
            per_device_eval_batch_size=MODEL_CONFIG['berto']['batch_size'],
            learning_rate=MODEL_CONFIG['berto']['learning_rate'],
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            logging_dir=MODELS_DIR / "logs",
            logging_steps=10,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=2,
            remove_unused_columns=False,
        )
        
        # Crear trainer
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if val_dataset else []
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
        
        # Entrenar
        self.trainer.train()
        self.is_trained = True
        
        print("✅ BETO entrenado correctamente")
        
        return self.trainer
    
    def predict(self, texts):
        """Realiza predicciones con BETO"""
        if not self.is_trained:
            raise ValueError("El modelo BETO no ha sido entrenado")
        
        # Preparar datos
        dataset = self.prepare_dataset(texts)
        
        # Realizar predicciones
        predictions = self.trainer.predict(dataset)
        sigmoid = torch.nn.Sigmoid()
        probabilities = sigmoid(torch.Tensor(predictions.predictions))
        
        # Convertir a etiquetas binarias
        y_pred = np.zeros(probabilities.shape)
        y_pred[np.where(probabilities >= 0.5)] = 1
        
        return y_pred.astype(int), probabilities.detach().numpy()
    
    def save_model(self, filename='berto_model'):
        """Guarda el modelo BETO entrenado"""
        if self.is_trained:
            save_path = MODELS_DIR / filename
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"✅ Modelo BETO guardado en: {save_path}")
    
    def load_model(self, model_path):
        """Carga un modelo BETO entrenado"""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.is_trained = True
            print(f"✅ Modelo BETO cargado desde: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo BETO: {e}")
            return False