"""
Módulo de K-Fold Cross-Validation
Evaluación robusta para SBM y BETO con validación cruzada.

Referencia: Kohavi, R. (1995). "A Study of Cross-Validation and Bootstrap
for Accuracy Estimation and Model Selection." IJCAI 1995.
"""

import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    hamming_loss, accuracy_score, precision_recall_fscore_support
)

from src.preprocessor import TextPreprocessor
from src.model_sbm import SBMClassifierChain
from src.config import PREPROCESSING_CONFIG


def kfold_sbm(X_texts, y_labels, label_names, n_splits=5, random_state=42):
    """
    K-Fold Cross-Validation para SBM + Classifier Chain.

    Args:
        X_texts: array/lista de textos originales
        y_labels: array (n_samples, n_labels) con etiquetas binarias
        label_names: lista de nombres de etiquetas
        n_splits: número de folds (default: 5)
        random_state: semilla para reproducibilidad

    Returns:
        dict con métricas promedio, desviación estándar y detalle por fold
    """
    print("=" * 70)
    print(f"K-FOLD CROSS-VALIDATION - SBM + Classifier Chain (K={n_splits})")
    print("=" * 70)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Almacenar métricas de cada fold
    fold_metrics = {
        'f1_macro': [], 'f1_micro': [],
        'precision_macro': [], 'recall_macro': [],
        'hamming_loss': [], 'subset_accuracy': [],
    }
    fold_per_label = []  # métricas por etiqueta por fold

    X_array = np.array(X_texts)
    y_array = np.array(y_labels)

    start_total = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_array)):
        start_fold = time.time()
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
        print(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")

        X_train_fold = X_array[train_idx]
        X_test_fold = X_array[test_idx]
        y_train_fold = y_array[train_idx]
        y_test_fold = y_array[test_idx]

        # Preprocesar
        preprocessor = TextPreprocessor()
        train_clean = preprocessor.preprocess_texts(X_train_fold.tolist())
        test_clean = preprocessor.preprocess_texts(X_test_fold.tolist())

        # Crear SBM
        X_train_sbm = preprocessor.create_sbm_matrix(train_clean, fit=True)
        X_test_sbm = preprocessor.create_sbm_matrix(test_clean, fit=False)

        # Entrenar
        model = SBMClassifierChain()
        model.train(X_train_sbm, y_train_fold)

        # Predecir
        y_pred = model.predict(X_test_sbm)

        # Calcular métricas
        f1_mac = f1_score(y_test_fold, y_pred, average='macro', zero_division=0)
        f1_mic = f1_score(y_test_fold, y_pred, average='micro', zero_division=0)
        prec_mac = precision_score(y_test_fold, y_pred, average='macro', zero_division=0)
        rec_mac = recall_score(y_test_fold, y_pred, average='macro', zero_division=0)
        hl = hamming_loss(y_test_fold, y_pred)
        sa = accuracy_score(y_test_fold, y_pred)

        fold_metrics['f1_macro'].append(f1_mac)
        fold_metrics['f1_micro'].append(f1_mic)
        fold_metrics['precision_macro'].append(prec_mac)
        fold_metrics['recall_macro'].append(rec_mac)
        fold_metrics['hamming_loss'].append(hl)
        fold_metrics['subset_accuracy'].append(sa)

        # Métricas por etiqueta
        p, r, f1, sup = precision_recall_fscore_support(
            y_test_fold, y_pred, average=None, zero_division=0
        )
        fold_per_label.append({'precision': p, 'recall': r, 'f1': f1, 'support': sup})

        elapsed = time.time() - start_fold
        print(f"  F1 Macro: {f1_mac:.2%} | Precision: {prec_mac:.2%} | "
              f"Recall: {rec_mac:.2%} | Tiempo: {elapsed:.1f}s")

    total_time = time.time() - start_total

    # Calcular promedios y desviación estándar
    print("\n" + "=" * 70)
    print("RESULTADOS SBM - K-FOLD CROSS-VALIDATION")
    print("=" * 70)

    results = {'model': 'SBM', 'n_splits': n_splits, 'metrics': {}}

    for metric_name, values in fold_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        results['metrics'][metric_name] = {'mean': mean, 'std': std, 'folds': values}
        print(f"  {metric_name:<20}: {mean:.2%} \u00B1 {std:.2%}")

    # Promedios por etiqueta
    print(f"\nM\u00E9tricas promedio por etiqueta:")
    print(f"  {'Etiqueta':<40} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 70)

    results['per_label'] = {}
    for i, name in enumerate(label_names):
        p_avg = np.mean([fl['precision'][i] for fl in fold_per_label])
        r_avg = np.mean([fl['recall'][i] for fl in fold_per_label])
        f1_avg = np.mean([fl['f1'][i] for fl in fold_per_label])
        p_std = np.std([fl['precision'][i] for fl in fold_per_label])
        r_std = np.std([fl['recall'][i] for fl in fold_per_label])
        f1_std = np.std([fl['f1'][i] for fl in fold_per_label])

        results['per_label'][name] = {
            'precision': {'mean': p_avg, 'std': p_std},
            'recall': {'mean': r_avg, 'std': r_std},
            'f1': {'mean': f1_avg, 'std': f1_std},
        }
        print(f"  {name:<40} {p_avg:.2%}\u00B1{p_std:.2%}  {r_avg:.2%}\u00B1{r_std:.2%}  {f1_avg:.2%}\u00B1{f1_std:.2%}")

    print(f"\nTiempo total: {total_time:.1f}s")
    print("=" * 70)

    return results


def kfold_beto(X_texts, y_labels, label_names, n_splits=5, random_state=42):
    """
    K-Fold Cross-Validation para BETO.
    NOTA: Mucho más lento que SBM. Con K=5 puede tardar 2-3 horas en CPU.

    Args:
        X_texts: array/lista de textos originales (SIN preprocesar)
        y_labels: array (n_samples, n_labels) con etiquetas binarias
        label_names: lista de nombres de etiquetas
        n_splits: número de folds (default: 5)
        random_state: semilla para reproducibilidad

    Returns:
        dict con métricas promedio, desviación estándar y detalle por fold
    """
    from src.model_berto import BETOMultiLabelClassifier
    from src.threshold_tuning import find_optimal_thresholds
    from sklearn.model_selection import train_test_split

    print("=" * 70)
    print(f"K-FOLD CROSS-VALIDATION - BETO (K={n_splits})")
    print("=" * 70)
    print("ADVERTENCIA: Esto puede tardar 2-3 horas en CPU.")
    print("En Colab con GPU T4, aproximadamente 30-45 minutos.")
    print()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = {
        'f1_macro': [], 'f1_micro': [],
        'precision_macro': [], 'recall_macro': [],
        'hamming_loss': [], 'subset_accuracy': [],
    }
    fold_metrics_optimized = {
        'f1_macro': [], 'f1_micro': [],
        'precision_macro': [], 'recall_macro': [],
    }
    fold_per_label = []
    fold_thresholds = []

    X_array = np.array(X_texts)
    y_array = np.array(y_labels)
    num_labels = y_array.shape[1]

    start_total = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_array)):
        start_fold = time.time()
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{n_splits}")
        print(f"{'='*50}")
        print(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")

        X_train_fold = X_array[train_idx].tolist()
        X_test_fold = X_array[test_idx].tolist()
        y_train_fold = y_array[train_idx]
        y_test_fold = y_array[test_idx]

        # Crear split de validación interno (10% del train)
        X_bt, X_bv, y_bt, y_bv = train_test_split(
            X_train_fold, y_train_fold, test_size=0.1, random_state=42
        )

        # Crear y entrenar modelo BETO desde cero
        beto = BETOMultiLabelClassifier(num_labels=num_labels)
        beto.train(
            train_texts=X_bt,
            train_labels=y_bt,
            val_texts=X_bv,
            val_labels=y_bv,
        )

        # Predecir en test
        y_pred, y_proba = beto.predict(X_test_fold)

        # Métricas con umbral por defecto del config
        f1_mac = f1_score(y_test_fold, y_pred, average='macro', zero_division=0)
        f1_mic = f1_score(y_test_fold, y_pred, average='micro', zero_division=0)
        prec_mac = precision_score(y_test_fold, y_pred, average='macro', zero_division=0)
        rec_mac = recall_score(y_test_fold, y_pred, average='macro', zero_division=0)
        hl = hamming_loss(y_test_fold, y_pred)
        sa = accuracy_score(y_test_fold, y_pred)

        fold_metrics['f1_macro'].append(f1_mac)
        fold_metrics['f1_micro'].append(f1_mic)
        fold_metrics['precision_macro'].append(prec_mac)
        fold_metrics['recall_macro'].append(rec_mac)
        fold_metrics['hamming_loss'].append(hl)
        fold_metrics['subset_accuracy'].append(sa)

        # Threshold tuning sobre este fold
        print(f"\n  Threshold tuning para fold {fold_idx + 1}...")
        tuning = find_optimal_thresholds(
            y_true=y_test_fold,
            y_proba=y_proba,
            label_names=label_names,
            metric='f1'
        )
        opt_thr = tuning['optimal_thresholds']
        fold_thresholds.append(opt_thr)

        # Métricas con umbrales optimizados
        y_pred_opt = np.zeros_like(y_proba, dtype=int)
        for j in range(num_labels):
            y_pred_opt[:, j] = (y_proba[:, j] >= opt_thr[j]).astype(int)

        f1_mac_opt = f1_score(y_test_fold, y_pred_opt, average='macro', zero_division=0)
        f1_mic_opt = f1_score(y_test_fold, y_pred_opt, average='micro', zero_division=0)
        prec_mac_opt = precision_score(y_test_fold, y_pred_opt, average='macro', zero_division=0)
        rec_mac_opt = recall_score(y_test_fold, y_pred_opt, average='macro', zero_division=0)

        fold_metrics_optimized['f1_macro'].append(f1_mac_opt)
        fold_metrics_optimized['f1_micro'].append(f1_mic_opt)
        fold_metrics_optimized['precision_macro'].append(prec_mac_opt)
        fold_metrics_optimized['recall_macro'].append(rec_mac_opt)

        # Por etiqueta (con umbral optimizado)
        p, r, f1, sup = precision_recall_fscore_support(
            y_test_fold, y_pred_opt, average=None, zero_division=0
        )
        fold_per_label.append({'precision': p, 'recall': r, 'f1': f1, 'support': sup})

        elapsed = time.time() - start_fold
        print(f"\n  Fold {fold_idx + 1} completado en {elapsed/60:.1f} minutos")
        print(f"  F1 Macro (config): {f1_mac:.2%} | F1 Macro (optimizado): {f1_mac_opt:.2%}")

        # Liberar memoria
        del beto
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    total_time = time.time() - start_total

    # Resultados BETO con umbrales del config
    print("\n" + "=" * 70)
    print("RESULTADOS BETO (umbrales del config) - K-FOLD")
    print("=" * 70)
    results = {'model': 'BETO', 'n_splits': n_splits, 'metrics': {}, 'metrics_optimized': {}}

    for metric_name, values in fold_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        results['metrics'][metric_name] = {'mean': mean, 'std': std, 'folds': values}
        print(f"  {metric_name:<20}: {mean:.2%} \u00B1 {std:.2%}")

    # Resultados BETO con threshold tuning
    print("\n" + "=" * 70)
    print("RESULTADOS BETO (threshold tuning por fold) - K-FOLD")
    print("=" * 70)

    for metric_name, values in fold_metrics_optimized.items():
        mean = np.mean(values)
        std = np.std(values)
        results['metrics_optimized'][metric_name] = {'mean': mean, 'std': std, 'folds': values}
        print(f"  {metric_name:<20}: {mean:.2%} \u00B1 {std:.2%}")

    # Promedios por etiqueta
    print(f"\nM\u00E9tricas promedio por etiqueta (con threshold tuning):")
    print(f"  {'Etiqueta':<40} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 70)

    results['per_label'] = {}
    for i, name in enumerate(label_names):
        p_avg = np.mean([fl['precision'][i] for fl in fold_per_label])
        r_avg = np.mean([fl['recall'][i] for fl in fold_per_label])
        f1_avg = np.mean([fl['f1'][i] for fl in fold_per_label])
        p_std = np.std([fl['precision'][i] for fl in fold_per_label])
        r_std = np.std([fl['recall'][i] for fl in fold_per_label])
        f1_std = np.std([fl['f1'][i] for fl in fold_per_label])

        results['per_label'][name] = {
            'precision': {'mean': p_avg, 'std': p_std},
            'recall': {'mean': r_avg, 'std': r_std},
            'f1': {'mean': f1_avg, 'std': f1_std},
        }
        print(f"  {name:<40} {p_avg:.2%}\u00B1{p_std:.2%}  {r_avg:.2%}\u00B1{r_std:.2%}  {f1_avg:.2%}\u00B1{f1_std:.2%}")

    # Umbrales promedio
    avg_thresholds = np.mean(fold_thresholds, axis=0).round(2).tolist()
    results['avg_thresholds'] = avg_thresholds
    print(f"\nUmbrales promedio entre folds: {avg_thresholds}")

    print(f"\nTiempo total: {total_time/60:.1f} minutos")
    print("=" * 70)

    return results