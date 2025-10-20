import pandas as pd
import joblib
from pathlib import Path
from .data_loader import DataLoader
from .preprocessor import TextPreprocessor
from .model_sbm import SBMClassifierChain
from .model_berto import BETOMultiLabelClassifier
from .evaluator import ModelEvaluator
from .config import DATA_DIR, MODELS_DIR

class ModelTrainer:
    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = TextPreprocessor()
        self.evaluator = None
        
    def setup_data(self):
        """Configura y prepara todos los datos"""
        print("📂 CONFIGURANDO DATOS...")
        
        # Cargar datos
        data = self.data_loader.load_data()
        if data is None:
            return False
        
        # Explorar datos
        text_col, label_cols = self.data_loader.explore_data()
        self.label_columns = label_cols
        
        # Dividir datos
        X_train, X_test, y_train, y_test = self.data_loader.split_data()
        
        # Preprocesar
        X_train_clean = self.preprocessor.preprocess_texts(X_train)
        X_test_clean = self.preprocessor.preprocess_texts(X_test)
        
        # Guardar datos procesados
        self.data_loader.save_processed_data(X_train_clean, X_test_clean, y_train, y_test)
        
        self.X_train = X_train_clean
        self.X_test = X_test_clean
        self.y_train = y_train.values
        self.y_test = y_test.values
        
        print(f"✅ Datos configurados: {len(self.X_train)} entrenamiento, {len(self.X_test)} prueba")
        return True
    
    def train_sbm_model(self):
        """Entrena el modelo SBM + Classifier Chain"""
        print("\n🤖 ENTRENANDO MODELO SBM...")
        
        # Crear matriz SBM
        X_train_sbm = self.preprocessor.create_sbm_matrix(self.X_train, fit=True)
        X_test_sbm = self.preprocessor.create_sbm_matrix(self.X_test, fit=False)
        
        # Entrenar modelo
        self.sbm_model = SBMClassifierChain(model_type='logistic_regression')
        self.sbm_model.train(X_train_sbm, self.y_train)
        
        # Guardar modelo y vectorizer
        self.sbm_model.save_model()
        self.preprocessor.save_vectorizer()
        
        # Evaluar
        y_pred_sbm = self.sbm_model.predict(X_test_sbm)
        
        if self.evaluator is None:
            self.evaluator = ModelEvaluator(self.label_columns)
        
        self.evaluator.calculate_metrics(self.y_test, y_pred_sbm, 'SBM_ClassifierChain')
        self.evaluator.print_metrics('SBM_ClassifierChain')
        
        return y_pred_sbm
    
    def train_berto_model(self):
        """Entrena el modelo BETO"""
        print("\n🤖 ENTRENANDO MODELO BETO...")
        
        # Crear y entrenar modelo BETO
        self.berto_model = BETOMultiLabelClassifier(num_labels=len(self.label_columns))
        
        # Dividir entrenamiento para validación
        from sklearn.model_selection import train_test_split
        X_bert_train, X_bert_val, y_bert_train, y_bert_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )
        
        # Entrenar BETO
        self.berto_model.train(
            train_texts=X_bert_train,
            train_labels=y_bert_train,
            val_texts=X_bert_val,
            val_labels=y_bert_val
        )
        
        # Guardar modelo
        self.berto_model.save_model()
        
        # Evaluar
        y_pred_berto, y_proba_berto = self.berto_model.predict(self.X_test)
        
        if self.evaluator is None:
            self.evaluator = ModelEvaluator(self.label_columns)
        
        self.evaluator.calculate_metrics(self.y_test, y_pred_berto, 'BETO')
        self.evaluator.print_metrics('BETO')
        
        return y_pred_berto
    
    def compare_models(self):
        """Compara todos los modelos entrenados"""
        if self.evaluator and len(self.evaluator.results) > 0:
            print("\n🔍 COMPARANDO MODELOS...")
            comparison = self.evaluator.compare_models()
            self.evaluator.plot_metrics_comparison()
            self.evaluator.save_results()
            return comparison
        else:
            print("❌ No hay modelos para comparar")
            return None
    
    def run_full_training(self):
        """Ejecuta el entrenamiento completo"""
        print("🚀 INICIANDO ENTRENAMIENTO COMPLETO")
        print("=" * 50)
        
        # Configurar datos
        if not self.setup_data():
            return
        
        # Entrenar modelos
        self.train_sbm_model()
        self.train_berto_model()
        
        # Comparar resultados
        self.compare_models()
        
        print("\n🎉 ENTRENAMIENTO COMPLETADO!")