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
from torch.utils.data import Dataset

class BETODataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        # número de ejemplos = tamaño de la primera dimensión de input_ids
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        # devuelve un dict {input_ids: ..., attention_mask: ..., labels: ...}
        return {k: v[idx] for k, v in self.encodings.items()}

class WeightedTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

class BETOMultiLabelClassifier:
    def __init__(self, num_labels):
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        self.is_trained = False
        # 🧠 Hiperparámetros desde la config
        berto_cfg = MODEL_CONFIG.get('berto', {})
        self.threshold = berto_cfg.get("threshold", 0.5)
        self.use_pos_weight = berto_cfg.get("use_pos_weight", True)
        self.pos_weight_cap = berto_cfg.get("pos_weight_cap", 10.0)
        
        # cuántas épocas, batch size y learning rate
        self.num_train_epochs = berto_cfg.get('epochs', 3)
        self.batch_size = berto_cfg.get('batch_size', 8)
        self.learning_rate = berto_cfg.get('learning_rate', 5e-5)

        # carpeta donde se guarda el modelo BETO
        # MODELS_DIR viene de config y es un Path
        self.output_dir = MODELS_DIR / berto_cfg.get('output_dir', 'berto_model')

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
            texts.tolist() if hasattr(texts, "tolist") else texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if labels is not None:
            encodings["labels"] = torch.tensor(labels, dtype=torch.float)
    
        # 🔁 Devolvemos un Dataset compatible con HuggingFace Trainer
        return BETODataset(encodings)
    
    
    def compute_metrics(self, eval_pred):
        """Calcula métricas para evaluación"""
        predictions, labels = eval_pred
        probs = torch.sigmoid(torch.Tensor(predictions)).numpy()

        # Soportar threshold como escalar o como lista/array por etiqueta
        thr = np.array(self.threshold, dtype=np.float32)
        if thr.ndim == 0:
            y_pred = (probs >= float(thr)).astype(int)
        else:
            y_pred = (probs >= thr.reshape(1, -1)).astype(int)

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
        # ---- pos_weight (balanceo) ----
        train_labels_np = np.array(train_labels)
        self.pos_weight = None
        if self.use_pos_weight:
            pos = train_labels_np.sum(axis=0).astype(np.float32)
            neg = (train_labels_np.shape[0] - pos).astype(np.float32)

            pos_weight_np = np.ones_like(pos, dtype=np.float32)
            mask = pos > 0
            pos_weight_np[mask] = neg[mask] / pos[mask]

            # cap configurable
            pos_weight_np = np.clip(pos_weight_np, 1.0, float(self.pos_weight_cap))

            self.pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32)

            print("🔍 Positivos por etiqueta en TRAIN:", pos.astype(int))
            print("✅ pos_weight:", pos_weight_np)
        else:
            print("ℹ️ pos_weight desactivado (Trainer estándar)")

        
        # Preparar datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        
        if val_texts is not None and val_labels is not None:
            val_dataset = self.prepare_dataset(val_texts, val_labels)
        else:
            val_dataset = None
        
        # Configurar argumentos de entrenamiento
        eval_strategy = "epoch" if val_dataset else "no"
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy=eval_strategy,
            save_strategy=eval_strategy,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1_macro" if val_dataset else None,
            greater_is_better=True if val_dataset else None,
            logging_dir="results/berto_logs",
            logging_steps=10,
            report_to=[],
        )


        
        # Crear trainer
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if val_dataset else []
        
        if self.use_pos_weight and self.pos_weight is not None:
            self.trainer = WeightedTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics,
                pos_weight=self.pos_weight,
                callbacks=callbacks,
            )
        else:
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
        """Realiza predicciones con BETO (sin depender de Trainer)"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("El modelo BETO no está cargado")

        # Si texts viene como lista o pandas series, lo convertimos a lista de strings
        texts_list = texts.tolist() if hasattr(texts, "tolist") else list(texts)

        # Tokenizar
        encodings = self.tokenizer(
            texts_list,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )

        # Enviar a dispositivo del modelo
        device = next(self.model.parameters()).device
        encodings = {k: v.to(device) for k, v in encodings.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

        # Threshold: puede ser float o lista/array por etiqueta
        thr = getattr(self, "threshold", 0.5)
        thr_arr = np.array(thr, dtype=np.float32)
        if thr_arr.ndim == 0:
            y_pred = (probs >= float(thr_arr)).astype(int)
        else:
            y_pred = (probs >= thr_arr.reshape(1, -1)).astype(int)

        return y_pred, probs
    
    def save_model(self, filename='berto_model'):
        """Guarda el modelo BETO entrenado"""
        if self.is_trained:
            save_path = self.output_dir  # ya es MODELS_DIR / 'berto_model' por defecto
            self.trainer.save_model(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))
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