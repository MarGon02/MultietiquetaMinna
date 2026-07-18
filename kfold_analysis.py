"""
Script de K-Fold Cross-Validation
Evaluación robusta de SBM y BETO.

Uso:
    python kfold_analysis.py          # Solo SBM (rápido, ~30 segundos)
    python kfold_analysis.py --beto   # SBM + BETO (lento, ~2-3 horas CPU)

Requisitos:
    Tener el dataset en data/raw/relatos_minna.csv
    NO necesita modelos preentrenados (entrena desde cero en cada fold)
"""

import sys
import numpy as np

from src.data_loader import DataLoader
from src.kfold_validation import kfold_sbm, kfold_beto

# ============================================================
# CONFIGURACIÓN
# ============================================================
N_SPLITS = 5           # Número de folds
RANDOM_STATE = 42      # Semilla para reproducibilidad
RUN_BETO = "--beto" in sys.argv  # Solo correr BETO si se pasa --beto


def main():
    print("=" * 70)
    print("K-FOLD CROSS-VALIDATION")
    print(f"K={N_SPLITS} folds | Semilla={RANDOM_STATE}")
    print("=" * 70)
    print()

    # 1. Cargar datos
    print("1. Cargando datos...")
    loader = DataLoader()
    df = loader.load_data()
    text_col, label_cols = loader.explore_data()

    X_texts = df[text_col].values
    y_labels = df[label_cols].values

    print(f"   Textos: {len(X_texts)}")
    print(f"   Etiquetas: {label_cols}")
    print()

    # 2. K-Fold SBM
    print("2. Ejecutando K-Fold para SBM...")
    results_sbm = kfold_sbm(
        X_texts=X_texts,
        y_labels=y_labels,
        label_names=label_cols,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )

    # 3. K-Fold BETO (opcional)
    results_beto = None
    if RUN_BETO:
        print("\n3. Ejecutando K-Fold para BETO...")
        results_beto = kfold_beto(
            X_texts=X_texts,
            y_labels=y_labels,
            label_names=label_cols,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
        )
    else:
        print("\n3. K-Fold BETO omitido (usar --beto para activarlo)")

    # 4. Resumen comparativo
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO FINAL")
    print("=" * 70)

    print(f"\n{'Métrica':<20} {'SBM (K-Fold)':>20}", end="")
    if results_beto:
        print(f" {'BETO config':>20} {'BETO optim.':>20}", end="")
    print()
    print("-" * (60 if not results_beto else 80))

    for metric in ['f1_macro', 'precision_macro', 'recall_macro', 'hamming_loss']:
        sbm_m = results_sbm['metrics'][metric]
        line = f"  {metric:<20} {sbm_m['mean']:.2%} \u00B1 {sbm_m['std']:.2%}"

        if results_beto:
            beto_m = results_beto['metrics'][metric]
            beto_o = results_beto['metrics_optimized'].get(metric)
            line += f"  {beto_m['mean']:.2%} \u00B1 {beto_m['std']:.2%}"
            if beto_o:
                line += f"  {beto_o['mean']:.2%} \u00B1 {beto_o['std']:.2%}"

        print(line)

    print()
    print("Interpretación:")
    print("  - El valor antes del \u00B1 es el promedio entre folds")
    print("  - El valor después del \u00B1 es la desviación estándar")
    print("  - Desviación baja (\u00B13-5%) = modelo estable")
    print("  - Desviación alta (\u00B115%+) = resultados dependen del split")
    print()

    if results_beto:
        print(f"Umbrales promedio BETO (entre folds): {results_beto['avg_thresholds']}")
        print("Estos umbrales se pueden usar en config.py como referencia.")


if __name__ == "__main__":
    main()