import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    hamming_loss, accuracy_score, classification_report,
    confusion_matrix
)
from pathlib import Path
from .config import RESULTS_DIR

class ModelEvaluator:
    def __init__(self, label_columns):
        self.label_columns = label_columns
        self.results = {}
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calcula todas las métricas para un modelo"""
        metrics = {
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'hamming_loss': hamming_loss(y_true, y_pred),
            'subset_accuracy': accuracy_score(y_true, y_pred)
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def print_metrics(self, model_name):
        """Imprime las métricas de un modelo"""
        if model_name in self.results:
            print(f"\n📊 MÉTRICAS - {model_name.upper()}")
            print("=" * 50)
            for metric, value in self.results[model_name].items():
                print(f"{metric:20}: {value:.4f}")
    
    def compare_models(self):
        """Compara todos los modelos evaluados"""
        if not self.results:
            print("No hay resultados para comparar")
            return
        
        comparison_df = pd.DataFrame(self.results).T
        print("\n🔍 COMPARACIÓN DE MODELOS")
        print("=" * 60)
        print(comparison_df.round(4))
        
        return comparison_df
    
    def plot_metrics_comparison(self):
        """Crea gráficos de comparación de modelos"""
        if len(self.results) < 2:
            print("Se necesitan al menos 2 modelos para comparar")
            return
        
        comparison_df = pd.DataFrame(self.results).T
        
        # Excluir hamming_loss para mejor visualización (es una métrica de pérdida)
        plot_metrics = [col for col in comparison_df.columns if col != 'hamming_loss']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(plot_metrics[:4]):
            comparison_df[metric].plot(kind='bar', ax=axes[i], color='skyblue')
            axes[i].set_title(f'Comparación de {metric}')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Guardar gráfico
        save_path = RESULTS_DIR / "metricas_comparacion.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {save_path}")
        
        plt.show()
    
    def save_results(self, filename='resultados_evaluacion.csv'):
        """Guarda los resultados en CSV"""
        if self.results:
            results_df = pd.DataFrame(self.results).T
            save_path = RESULTS_DIR / filename
            results_df.to_csv(save_path)
            print(f"✅ Resultados guardados en: {save_path}")
            return save_path