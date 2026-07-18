"""
Evaluación Robusta: Hold-out + K-Fold Cross-Validation
Metodología sin fuga de información (Varma & Simon, 2006).

Flujo:
  1. Separar 20% como TEST FINAL (no se toca hasta el final)
  2. K-Fold (K=5) sobre el 80% restante (desarrollo)
  3. Entrenar modelo FINAL con el 80% completo
  4. Evaluar sobre el 20% reservado → métricas definitivas

Uso:
    python kfold_analysis_v2.py          # Solo SBM (~1 segundo)
    python kfold_analysis_v2.py --beto   # SBM + BETO (~100 min con GPU)
"""

import sys
import time
import numpy as np
import gc
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    hamming_loss, accuracy_score, precision_recall_fscore_support
)

from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.model_sbm import SBMClassifierChain
from src.threshold_tuning import find_optimal_thresholds
from src.config import PREPROCESSING_CONFIG

# ============================================================
N_SPLITS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
RUN_BETO = "--beto" in sys.argv
# ============================================================


def calculate_metrics(y_true, y_pred):
    """Calcula todas las métricas de una vez."""
    return {
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'hamming_loss': hamming_loss(y_true, y_pred),
        'subset_accuracy': accuracy_score(y_true, y_pred),
    }


def print_metrics_table(metrics_dict, title):
    """Imprime métricas con formato."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    for name, val in metrics_dict.items():
        print(f"  {name:<20}: {val:.2%}")


def print_kfold_results(fold_metrics, title):
    """Imprime resultados de K-Fold con promedio ± std."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    results = {}
    for metric_name in fold_metrics[0].keys():
        values = [fm[metric_name] for fm in fold_metrics]
        mean = np.mean(values)
        std = np.std(values)
        results[metric_name] = {'mean': mean, 'std': std}
        print(f"  {metric_name:<20}: {mean:.2%} \u00B1 {std:.2%}")
    return results


def print_per_label(fold_per_label, label_names, title):
    """Imprime métricas promedio por etiqueta."""
    print(f"\n{title}")
    print(f"  {'Etiqueta':<40} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("  " + "-" * 76)
    results = {}
    for i, name in enumerate(label_names):
        p_vals = [fl['precision'][i] for fl in fold_per_label]
        r_vals = [fl['recall'][i] for fl in fold_per_label]
        f1_vals = [fl['f1'][i] for fl in fold_per_label]
        results[name] = {
            'precision': {'mean': np.mean(p_vals), 'std': np.std(p_vals)},
            'recall': {'mean': np.mean(r_vals), 'std': np.std(r_vals)},
            'f1': {'mean': np.mean(f1_vals), 'std': np.std(f1_vals)},
        }
        print(f"  {name:<40} {np.mean(p_vals):.2%}\u00B1{np.std(p_vals):.2%}"
              f"  {np.mean(r_vals):.2%}\u00B1{np.std(r_vals):.2%}"
              f"  {np.mean(f1_vals):.2%}\u00B1{np.std(f1_vals):.2%}")
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("EVALUACI\u00D3N ROBUSTA: HOLD-OUT + K-FOLD CROSS-VALIDATION")
    print(f"Test final: {int(TEST_SIZE*100)}% | K-Fold: K={N_SPLITS} | Semilla: {RANDOM_STATE}")
    print("=" * 70)

    # ========================================================
    # PASO 1: Cargar datos y separar TEST FINAL
    # ========================================================
    print("\n\u2501" * 70)
    print("PASO 1: CARGA DE DATOS Y SEPARACI\u00D3N DEL TEST FINAL")
    print("\u2501" * 70)

    loader = DataLoader()
    df = loader.load_data()
    text_col, label_cols = loader.explore_data()

    X_all = df[text_col].values
    y_all = df[label_cols].values

    # Separar test final (20%) - NO SE TOCA HASTA EL FINAL
    X_dev, X_test_final, y_dev, y_test_final = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"\n  Dataset total:    {len(X_all)} relatos")
    print(f"  Desarrollo (80%): {len(X_dev)} relatos \u2192 para K-Fold")
    print(f"  Test final (20%): {len(X_test_final)} relatos \u2192 NO se toca hasta el final")
    print(f"\n  Positivos en test final por etiqueta:")
    for i, name in enumerate(label_cols):
        print(f"    {name}: {int(y_test_final[:, i].sum())}")

    # ========================================================
    # PASO 2: K-FOLD SBM sobre desarrollo (240)
    # ========================================================
    print(f"\n\u2501" * 70)
    print("PASO 2: K-FOLD CROSS-VALIDATION - SBM (sobre desarrollo)")
    print("\u2501" * 70)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    sbm_fold_metrics = []
    sbm_fold_per_label = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_dev)):
        print(f"\n  --- Fold {fold_idx+1}/{N_SPLITS} (train:{len(train_idx)} val:{len(val_idx)}) ---")

        X_train_f = X_dev[train_idx]
        X_val_f = X_dev[val_idx]
        y_train_f = y_dev[train_idx]
        y_val_f = y_dev[val_idx]

        preprocessor = TextPreprocessor()
        train_clean = preprocessor.preprocess_texts(X_train_f.tolist())
        val_clean = preprocessor.preprocess_texts(X_val_f.tolist())

        X_train_sbm = preprocessor.create_sbm_matrix(train_clean, fit=True)
        X_val_sbm = preprocessor.create_sbm_matrix(val_clean, fit=False)

        model = SBMClassifierChain()
        model.train(X_train_sbm, y_train_f)
        y_pred = model.predict(X_val_sbm)

        metrics = calculate_metrics(y_val_f, y_pred)
        sbm_fold_metrics.append(metrics)

        pr, rc, f1, sup = precision_recall_fscore_support(y_val_f, y_pred, average=None, zero_division=0)
        sbm_fold_per_label.append({'precision': pr, 'recall': rc, 'f1': f1, 'support': sup})

        print(f"    F1 Macro: {metrics['f1_macro']:.2%} | Precision: {metrics['precision_macro']:.2%} | Recall: {metrics['recall_macro']:.2%}")

    sbm_kfold_results = print_kfold_results(sbm_fold_metrics, "RESULTADOS K-FOLD: SBM")
    sbm_kfold_per_label = print_per_label(sbm_fold_per_label, label_cols, "M\u00E9tricas por etiqueta (SBM K-Fold):")

    # ========================================================
    # PASO 3: K-FOLD BETO sobre desarrollo (240)
    # ========================================================
    beto_kfold_results = None
    beto_kfold_opt_results = None
    avg_thresholds = None

    if RUN_BETO:
        from src.model_berto import BETOMultiLabelClassifier
        import torch

        print(f"\n\u2501" * 70)
        print("PASO 3: K-FOLD CROSS-VALIDATION - BETO (sobre desarrollo)")
        print("\u2501" * 70)
        print("NOTA: ~100 minutos con GPU en Colab\n")

        beto_fold_metrics = []
        beto_fold_metrics_opt = []
        beto_fold_per_label = []
        beto_fold_thresholds = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_dev)):
            start = time.time()
            print(f"\n  {'='*50}")
            print(f"  Fold {fold_idx+1}/{N_SPLITS} (train:{len(train_idx)} val:{len(val_idx)})")
            print(f"  {'='*50}")

            X_train_f = X_dev[train_idx].tolist()
            X_val_f = X_dev[val_idx].tolist()
            y_train_f = y_dev[train_idx]
            y_val_f = y_dev[val_idx]

            # Split interno para validación de BETO (10% del train del fold)
            X_bt, X_bv, y_bt, y_bv = train_test_split(
                X_train_f, y_train_f, test_size=0.1, random_state=42
            )

            beto = BETOMultiLabelClassifier(num_labels=y_all.shape[1])
            beto.train(train_texts=X_bt, train_labels=y_bt, val_texts=X_bv, val_labels=y_bv)

            y_pred_b, y_proba_b = beto.predict(X_val_f)

            # Métricas con umbrales del config
            metrics_cfg = calculate_metrics(y_val_f, y_pred_b)
            beto_fold_metrics.append(metrics_cfg)

            # Threshold tuning sobre el fold de validación
            print(f"\n  Threshold tuning fold {fold_idx+1}...")
            tuning = find_optimal_thresholds(y_val_f, y_proba_b, label_cols, metric='f1')
            opt_thr = tuning['optimal_thresholds']
            beto_fold_thresholds.append(opt_thr)

            # Métricas con umbrales optimizados
            y_pred_opt = np.zeros_like(y_proba_b, dtype=int)
            for j in range(y_all.shape[1]):
                y_pred_opt[:, j] = (y_proba_b[:, j] >= opt_thr[j]).astype(int)

            metrics_opt = calculate_metrics(y_val_f, y_pred_opt)
            beto_fold_metrics_opt.append(metrics_opt)

            pr, rc, f1, sup = precision_recall_fscore_support(y_val_f, y_pred_opt, average=None, zero_division=0)
            beto_fold_per_label.append({'precision': pr, 'recall': rc, 'f1': f1, 'support': sup})

            elapsed = time.time() - start
            print(f"\n  Fold {fold_idx+1} completado en {elapsed/60:.1f} min")
            print(f"  F1 config: {metrics_cfg['f1_macro']:.2%} | F1 optim: {metrics_opt['f1_macro']:.2%}")

            del beto
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        beto_kfold_results = print_kfold_results(beto_fold_metrics, "RESULTADOS K-FOLD: BETO (umbrales config)")
        beto_kfold_opt_results = print_kfold_results(beto_fold_metrics_opt, "RESULTADOS K-FOLD: BETO (threshold tuning)")
        beto_kfold_per_label = print_per_label(beto_fold_per_label, label_cols, "M\u00E9tricas por etiqueta (BETO optimizado K-Fold):")

        avg_thresholds = np.mean(beto_fold_thresholds, axis=0).round(2).tolist()
        print(f"\n  Umbrales promedio entre folds: {avg_thresholds}")
    else:
        print(f"\n  PASO 3: BETO omitido (usar --beto para activar)")

    # ========================================================
    # PASO 4: ENTRENAR MODELOS FINALES con los 240 completos
    # ========================================================
    print(f"\n\u2501" * 70)
    print("PASO 4: ENTRENAMIENTO DE MODELOS FINALES (240 datos completos)")
    print("\u2501" * 70)

    # --- SBM FINAL ---
    print("\n  Entrenando SBM final...")
    preprocessor_final = TextPreprocessor()
    dev_clean = preprocessor_final.preprocess_texts(X_dev.tolist())
    X_dev_sbm = preprocessor_final.create_sbm_matrix(dev_clean, fit=True)

    sbm_final = SBMClassifierChain()
    sbm_final.train(X_dev_sbm, y_dev)
    print("  SBM final entrenado")

    # --- BETO FINAL ---
    beto_final = None
    if RUN_BETO:
        from src.model_berto import BETOMultiLabelClassifier
        import torch

        print("\n  Entrenando BETO final (esto puede tardar ~20 min)...")

        X_bt, X_bv, y_bt, y_bv = train_test_split(
            X_dev.tolist(), y_dev, test_size=0.1, random_state=42
        )

        beto_final = BETOMultiLabelClassifier(num_labels=y_all.shape[1])
        beto_final.train(train_texts=X_bt, train_labels=y_bt, val_texts=X_bv, val_labels=y_bv)

        # Usar umbrales promedio del K-Fold
        if avg_thresholds:
            beto_final.threshold = avg_thresholds
            print(f"  Umbrales aplicados: {avg_thresholds}")
        print("  BETO final entrenado")

    # ========================================================
    # PASO 5: EVALUACI\u00D3N SOBRE TEST FINAL (60 datos que nunca se tocaron)
    # ========================================================
    print(f"\n\u2501" * 70)
    print("PASO 5: EVALUACI\u00D3N SOBRE TEST FINAL (datos nunca vistos)")
    print("\u2501" * 70)
    print(f"  Evaluando sobre {len(X_test_final)} relatos reservados desde el inicio")

    # --- SBM sobre test final ---
    test_clean = preprocessor_final.preprocess_texts(X_test_final.tolist())
    X_test_sbm = preprocessor_final.create_sbm_matrix(test_clean, fit=False)
    y_pred_sbm_final = sbm_final.predict(X_test_sbm)

    sbm_final_metrics = calculate_metrics(y_test_final, y_pred_sbm_final)
    print_metrics_table(sbm_final_metrics, "TEST FINAL: SBM")

    pr, rc, f1, sup = precision_recall_fscore_support(y_test_final, y_pred_sbm_final, average=None, zero_division=0)
    print(f"\n  M\u00E9tricas por etiqueta (SBM - Test Final):")
    for i, name in enumerate(label_cols):
        print(f"    {name}: P={pr[i]:.2%} R={rc[i]:.2%} F1={f1[i]:.2%} (support={int(sup[i])})")

    # --- BETO sobre test final ---
    beto_final_metrics = None
    if beto_final is not None:
        y_pred_beto_final, y_proba_beto_final = beto_final.predict(X_test_final.tolist())

        beto_final_metrics = calculate_metrics(y_test_final, y_pred_beto_final)
        print_metrics_table(beto_final_metrics, "TEST FINAL: BETO (umbrales promedio K-Fold)")

        pr, rc, f1, sup = precision_recall_fscore_support(y_test_final, y_pred_beto_final, average=None, zero_division=0)
        print(f"\n  M\u00E9tricas por etiqueta (BETO - Test Final):")
        for i, name in enumerate(label_cols):
            print(f"    {name}: P={pr[i]:.2%} R={rc[i]:.2%} F1={f1[i]:.2%} (support={int(sup[i])})")

        print(f"\n  Umbrales usados: {avg_thresholds}")

    # ========================================================
    # PASO 6: RESUMEN COMPARATIVO COMPLETO
    # ========================================================
    print(f"\n{'='*70}")
    print("RESUMEN COMPARATIVO COMPLETO")
    print(f"{'='*70}")

    print(f"\n--- Fase de Desarrollo (K-Fold sobre 240 datos) ---")
    print(f"{'M\u00E9trica':<20} {'SBM':>20}", end="")
    if beto_kfold_opt_results:
        print(f" {'BETO optim.':>20}", end="")
    print()
    print("-" * (40 if not beto_kfold_opt_results else 60))

    for m in ['f1_macro', 'precision_macro', 'recall_macro']:
        sbm_r = sbm_kfold_results[m]
        line = f"  {m:<20} {sbm_r['mean']:.2%} \u00B1 {sbm_r['std']:.2%}"
        if beto_kfold_opt_results:
            beto_r = beto_kfold_opt_results[m]
            line += f"  {beto_r['mean']:.2%} \u00B1 {beto_r['std']:.2%}"
        print(line)

    print(f"\n--- Evaluaci\u00F3n Final (Test de 60 datos nunca vistos) ---")
    print(f"{'M\u00E9trica':<20} {'SBM':>20}", end="")
    if beto_final_metrics:
        print(f" {'BETO optim.':>20}", end="")
    print()
    print("-" * (40 if not beto_final_metrics else 60))

    for m in ['f1_macro', 'precision_macro', 'recall_macro', 'hamming_loss']:
        line = f"  {m:<20} {sbm_final_metrics[m]:.2%}"
        if beto_final_metrics:
            line += f"               {beto_final_metrics[m]:.2%}"
        print(line)

    print(f"\n{'='*70}")
    print("INTERPRETACI\u00D3N")
    print(f"{'='*70}")
    print("  Las m\u00E9tricas del TEST FINAL son las definitivas para el informe.")
    print("  No hubo fuga de informaci\u00F3n: los umbrales se optimizaron en K-Fold")
    print("  (desarrollo) y se evaluaron sobre datos completamente nuevos.")
    print("  Referencia: Varma & Simon (2006), Cawley & Talbot (2010).")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()