import sys
from pathlib import Path
import numpy as np
# Agregar src al path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))
# cambio de parametro de umbrall para beto
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.model_sbm import SBMClassifierChain
from src.evaluator import ModelEvaluator
from src.model_berto import BETOMultiLabelClassifier 

def main():
    print("🚀 INICIANDO PROYECTO PLN - MINNA")
    print("=" * 50)
    
    # 1. Cargar datos
    print("\n1. 📂 CARGANDO DATOS...")
    loader = DataLoader()
    data = loader.load_data()
    
    if data is None:
        print("❌ No se pudieron cargar los datos. Saliendo...")
        return
    
    # 2. Explorar datos
    print("\n2. 🔍 EXPLORANDO DATOS...")
    text_col, label_cols = loader.explore_data()
    
    # 3. Dividir datos
    print("\n3. 📊 DIVIDIENDO DATOS...")
    X_train, X_test, y_train, y_test = loader.split_data()
    
    # 4. Preprocesar textos
    print("\n4. 🧹 PREPROCESANDO TEXTOS...")
    preprocessor = TextPreprocessor()
    
    X_train_clean = preprocessor.preprocess_texts(X_train)
    X_test_clean = preprocessor.preprocess_texts(X_test)
    
    # 5. Crear matrices SBM
    print("\n5. 📈 CREANDO MATRICES SBM...")
    X_train_sbm = preprocessor.create_sbm_matrix(X_train_clean, fit=True)
    X_test_sbm = preprocessor.create_sbm_matrix(X_test_clean, fit=False)
    
    # Guardar vectorizer
    preprocessor.save_vectorizer()
    
    # 6. Entrenar modelo SBM
    print("\n6. 🤖 ENTRENANDO MODELO SBM + CLASSIFIER CHAIN...")
    sbm_model = SBMClassifierChain(model_type='logistic_regression')
    sbm_model.train(X_train_sbm, y_train)
    
    # Guardar modelo
    sbm_model.save_model()
    
    # 7. Evaluar modelo
    print("\n7. 📊 EVALUANDO MODELO...")
    evaluator = ModelEvaluator(label_cols)
    
    # Predecir y evaluar
    y_pred_sbm = sbm_model.predict(X_test_sbm)
    metrics_sbm = evaluator.calculate_metrics(y_test, y_pred_sbm, 'SBM_ClassifierChain')
    
    # Mostrar resultados
    evaluator.print_metrics('SBM_ClassifierChain')

     # 7.b Entrenar y evaluar modelo BETO
    print("\n7.b 🤖 ENTRENANDO MODELO BETO...")
    from sklearn.model_selection import train_test_split

    # Convertir etiquetas a numpy para BETO
    y_train_array = y_train.values
    y_test_array = y_test.values

    # Crear modelo BETO
    berto_model = BETOMultiLabelClassifier(num_labels=len(label_cols))

    # Dividir entrenamiento en train/validación para BETO
    # IMPORTANTE: usar textos ORIGINALES (X_train), no los limpios.
    # BETO tiene su propio tokenizer y se beneficia del texto completo.
    X_train_raw = X_train.tolist()
    X_test_raw = X_test.tolist()

    X_bert_train, X_bert_val, y_bert_train, y_bert_val = train_test_split(
        X_train_raw, y_train_array, test_size=0.1, random_state=42
    )

    # Entrenar BETO
    berto_model.train(
        train_texts=X_bert_train,
        train_labels=y_bert_train,
        val_texts=X_bert_val,
        val_labels=y_bert_val
    )

    # Guardar modelo BETO
    berto_model.save_model()

    # Evaluar BETO en el conjunto de test
    print("\n7.c 📊 EVALUANDO MODELO BETO...")
    y_pred_berto, y_proba_berto = berto_model.predict(X_test_raw)
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f1, support = precision_recall_fscore_support(
        y_test_array, y_pred_berto, average=None, zero_division=0
    )

    print("\n📌 MÉTRICAS POR ETIQUETA (BETO)")
    for i, name in enumerate(label_cols):
        print(f"- {name}: precision={p[i]:.3f} | recall={r[i]:.3f} | f1={f1[i]:.3f} | support={support[i]}")

    evaluator.calculate_metrics(y_test_array, y_pred_berto, 'BETO')
    evaluator.print_metrics('BETO')
    print("🔍 Positivos predichos por etiqueta:", y_pred_berto.sum(axis=0))
    print("🔍 Total ejemplos test:", y_test_array.shape[0])

    idx = 4  # columna negligencia
    pos_idx = np.where(y_test_array[:, idx] == 1)[0]
    print("Indices con negligencia real=1:", pos_idx)

    if len(pos_idx) > 0:
        i = pos_idx[0]
        print("Probabilidad BETO para el caso real de negligencia:", y_proba_berto[i, idx])
        print("Predicción BETO para ese caso:", y_pred_berto[i, idx])
    
    # 8. Guardar resultados
    print("\n8. 💾 GUARDANDO RESULTADOS...")
    evaluator.save_results()
    evaluator.plot_metrics_comparison()
    
    # Guardar datos procesados
    loader.save_processed_data(X_train_clean, X_test_clean, y_train, y_test)
    
    print("\n🎉 PROCESO COMPLETADO EXITOSAMENTE!")
    print("Puedes encontrar todos los archivos guardados en sus respectivas carpetas.")

if __name__ == "__main__":
    main()
