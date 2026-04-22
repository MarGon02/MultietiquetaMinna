"""
Módulo de Threshold Tuning Automático
Encuentra el umbral óptimo por etiqueta que maximiza F1-Score
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def find_optimal_thresholds(y_true, y_proba, label_names=None, metric='f1'):
    """
    Busca el umbral óptimo para cada etiqueta que maximiza el F1-Score.
    
    Args:
        y_true: array (n_samples, n_labels) con etiquetas reales (0/1)
        y_proba: array (n_samples, n_labels) con probabilidades predichas
        label_names: lista de nombres de etiquetas (opcional)
        metric: métrica a optimizar ('f1', 'recall', 'precision')
    
    Returns:
        dict con resultados por etiqueta y umbrales óptimos
    """
    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"label_{i}" for i in range(n_labels)]
    
    # Rango de umbrales a probar
    thresholds = np.arange(0.05, 0.95, 0.01)
    
    optimal_thresholds = []
    results = []
    
    print("=" * 70)
    print("THRESHOLD TUNING AUTOMÁTICO")
    print("=" * 70)
    
    for i in range(n_labels):
        best_threshold = 0.50
        best_score = 0.0
        best_precision = 0.0
        best_recall = 0.0
        
        # Contar positivos reales
        n_positivos = int(y_true[:, i].sum())
        
        for thr in thresholds:
            y_pred_i = (y_proba[:, i] >= thr).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true[:, i], y_pred_i, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true[:, i], y_pred_i, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true[:, i], y_pred_i, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = round(float(thr), 2)
                best_precision = precision_score(y_true[:, i], y_pred_i, zero_division=0)
                best_recall = recall_score(y_true[:, i], y_pred_i, zero_division=0)
        
        optimal_thresholds.append(best_threshold)
        
        result = {
            'etiqueta': label_names[i],
            'umbral_optimo': best_threshold,
            'f1': best_score,
            'precision': best_precision,
            'recall': best_recall,
            'positivos_reales': n_positivos,
        }
        results.append(result)
        
        # Comparar con umbral fijo 0.50
        y_pred_050 = (y_proba[:, i] >= 0.50).astype(int)
        f1_050 = f1_score(y_true[:, i], y_pred_050, zero_division=0)
        mejora = best_score - f1_050
        
        print(f"\n{label_names[i]}:")
        print(f"  Positivos en test: {n_positivos}")
        print(f"  Umbral fijo 0.50  → F1={f1_050:.3f}")
        print(f"  Umbral óptimo {best_threshold:.2f} → F1={best_score:.3f} "
              f"(P={best_precision:.3f}, R={best_recall:.3f})")
        if mejora > 0:
            print(f"  Mejora: +{mejora:.3f} ↑")
        elif mejora == 0:
            print(f"  Sin cambio")
        else:
            print(f"  Diferencia: {mejora:.3f}")
    
    print("\n" + "=" * 70)
    print("RESUMEN DE UMBRALES ÓPTIMOS")
    print("=" * 70)
    print(f"Umbrales: {optimal_thresholds}")
    print(f"Para config.py: 'threshold': {optimal_thresholds}")
    
    # Calcular F1 macro con umbrales óptimos vs 0.50
    y_pred_optimal = np.zeros_like(y_proba, dtype=int)
    y_pred_050 = (y_proba >= 0.50).astype(int)
    
    for i in range(n_labels):
        y_pred_optimal[:, i] = (y_proba[:, i] >= optimal_thresholds[i]).astype(int)
    
    f1_macro_optimal = f1_score(y_true, y_pred_optimal, average='macro', zero_division=0)
    f1_macro_050 = f1_score(y_true, y_pred_050, average='macro', zero_division=0)
    
    print(f"\nF1 Macro con umbral fijo 0.50:     {f1_macro_050:.4f}")
    print(f"F1 Macro con umbrales óptimos:     {f1_macro_optimal:.4f}")
    print(f"Mejora total:                      +{f1_macro_optimal - f1_macro_050:.4f}")
    print("=" * 70)
    
    return {
        'optimal_thresholds': optimal_thresholds,
        'results_per_label': results,
        'f1_macro_optimal': f1_macro_optimal,
        'f1_macro_050': f1_macro_050,
    }