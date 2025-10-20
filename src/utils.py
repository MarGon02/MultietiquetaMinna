import os
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

def setup_logging(log_dir="logs"):
    """Configura el sistema de logging"""
    Path(log_dir).mkdir(exist_ok=True)
    
    log_filename = f"pln_minna_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = Path(log_dir) / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def save_config(config, filename="training_config.json"):
    """Guarda la configuración del entrenamiento"""
    config_path = Path("configs") / filename
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Configuración guardada en: {config_path}")

def load_config(filename="training_config.json"):
    """Carga la configuración del entrenamiento"""
    config_path = Path("configs") / filename
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Configuración cargada desde: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Archivo de configuración no encontrado: {config_path}")
        return None

def analyze_predictions(y_true, y_pred, label_names):
    """Analiza las predicciones y genera reportes detallados"""
    from sklearn.metrics import classification_report, confusion_matrix
    
    analysis = {}
    
    # Reporte de clasificación por etiqueta
    analysis['classification_report'] = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
    
    # Matrices de confusión por etiqueta
    analysis['confusion_matrices'] = {}
    for i, label in enumerate(label_names):
        analysis['confusion_matrices'][label] = confusion_matrix(
            y_true[:, i], y_pred[:, i]
        ).tolist()
    
    # Análisis de multi-etiquetado
    analysis['multi_label_stats'] = {
        'avg_labels_per_instance_true': y_true.sum(axis=1).mean(),
        'avg_labels_per_instance_pred': y_pred.sum(axis=1).mean(),
        'total_labels_true': y_true.sum(),
        'total_labels_pred': y_pred.sum()
    }
    
    return analysis

def export_predictions(texts, y_true, y_pred, label_names, filename="predictions_analysis.xlsx"):
    """Exporta las predicciones a Excel para análisis manual"""
    results_df = pd.DataFrame({
        'texto': texts,
        'texto_length': [len(str(text)) for text in texts]
    })
    
    # Agregar etiquetas verdaderas y predichas
    for i, label in enumerate(label_names):
        results_df[f'true_{label}'] = y_true[:, i]
        results_df[f'pred_{label}'] = y_pred[:, i]
        results_df[f'correct_{label}'] = y_true[:, i] == y_pred[:, i]
    
    # Calcular precisión por instancia
    results_df['correct_all_labels'] = (y_true == y_pred).all(axis=1)
    results_df['num_correct_labels'] = (y_true == y_pred).sum(axis=1)
    
    # Guardar en Excel
    save_path = Path("results") / filename
    save_path.parent.mkdir(exist_ok=True)
    
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Resumen estadístico
        summary_data = []
        for label in label_names:
            accuracy = (results_df[f'true_{label}'] == results_df[f'pred_{label}']).mean()
            summary_data.append({
                'Label': label,
                'Accuracy': accuracy,
                'True_Positive': results_df[(results_df[f'true_{label}'] == 1) & (results_df[f'pred_{label}'] == 1)].shape[0],
                'False_Positive': results_df[(results_df[f'true_{label}'] == 0) & (results_df[f'pred_{label}'] == 1)].shape[0],
                'False_Negative': results_df[(results_df[f'true_{label}'] == 1) & (results_df[f'pred_{label}'] == 0)].shape[0]
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✅ Predicciones exportadas a: {save_path}")
    return save_path