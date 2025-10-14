import joblib
from pathlib import Path
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.config import MODELS_DIR, MODEL_CONFIG

class SBMClassifierChain:
    def __init__(self, model_type='logistic_regression'):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
        # Configurar el clasificador base
        if model_type == 'logistic_regression':
            base_estimator = LogisticRegression(
                random_state=MODEL_CONFIG['classifier_chain']['random_state'],
                max_iter=MODEL_CONFIG['classifier_chain']['max_iter']
            )
        elif model_type == 'random_forest':
            base_estimator = RandomForestClassifier(
                random_state=MODEL_CONFIG['classifier_chain']['random_state'],
                n_estimators=100
            )
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        # Crear Classifier Chain
        self.model = ClassifierChain(
            base_estimator,
            order='random',
            random_state=MODEL_CONFIG['classifier_chain']['random_state']
        )
    
    def train(self, X_train, y_train):
        """Entrena el modelo"""
        print("⏳ Entrenando Classifier Chain...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("✅ Classifier Chain entrenado")
        return self
    
    def predict(self, X):
        """Realiza predicciones"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Obtiene probabilidades de predicción"""
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        return self.model.predict_proba(X)
    
    def save_model(self, filename='classifier_chain_model.joblib'):
        """Guarda el modelo entrenado"""
        if self.is_trained:
            save_path = MODELS_DIR / filename
            joblib.dump(self.model, save_path)
            print(f"✅ Modelo Classifier Chain guardado en: {save_path}")
    
    def load_model(self, filename='classifier_chain_model.joblib'):
        """Carga un modelo entrenado"""
        load_path = MODELS_DIR / filename
        try:
            self.model = joblib.load(load_path)
            self.is_trained = True
            print(f"✅ Modelo Classifier Chain cargado desde: {load_path}")
            return True
        except FileNotFoundError:
            print(f"❌ Modelo no encontrado: {load_path}")
            return False