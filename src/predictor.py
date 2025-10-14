import pandas as pd
import joblib
from pathlib import Path
from src.preprocessor import TextPreprocessor
from src.model_sbm import SBMClassifierChain
from src.config import MODELS_DIR

class Predictor:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.sbm_model = SBMClassifierChain()
        self.label_columns = None
        self.is_loaded = False
    
    def load_models(self, vectorizer_file='sbm_vectorizer.joblib', 
                   model_file='classifier_chain_model.joblib',
                   labels_file='processed_data.joblib'):
        """Carga todos los modelos necesarios para predicciones"""
        # Cargar vectorizer
        if not self.preprocessor.load_vectorizer(vectorizer_file):
            return False
        
        # Cargar modelo
        if not self.sbm_model.load_model(model_file):
            return False
        
        # Cargar etiquetas
        try:
            from .config import DATA_DIR
            processed_data = joblib.load(DATA_DIR / "processed" / labels_file)
            self.label_columns = processed_data['label_columns']
            print(f"✅ Etiquetas cargadas: {len(self.label_columns)} etiquetas")
        except FileNotFoundError:
            print("⚠️  No se pudieron cargar las etiquetas, usando predeterminadas")
            self.label_columns = ['violencia_fisica', 'violencia_psicologica', 'abandono', 'abuso_sexual']
        
        self.is_loaded = True
        print("✅ Todos los modelos cargados correctamente")
        return True
    
    def predict_single_text(self, text):
        """Predice etiquetas para un solo texto"""
        if not self.is_loaded:
            print("❌ Los modelos no están cargados. Ejecuta load_models() primero.")
            return None
        
        # Preprocesar texto
        cleaned_text = self.preprocessor.preprocess_texts([text])[0]
        
        # Crear SBM
        text_sbm = self.preprocessor.create_sbm_matrix([cleaned_text], fit=False)
        
        # Predecir
        prediction = self.sbm_model.predict(text_sbm)
        probabilities = self.sbm_model.predict_proba(text_sbm)
        
        # Formatear resultados
        result = {
            'texto': text,
            'texto_limpio': cleaned_text,
            'etiquetas_predichas': prediction[0],
            'probabilidades': probabilities[0]
        }
        
        return result
    
    def predict_batch(self, texts):
        """Predice etiquetas para un lote de textos"""
        if not self.is_loaded:
            print("❌ Los modelos no están cargados. Ejecuta load_models() primero.")
            return None
        
        # Preprocesar textos
        cleaned_texts = self.preprocessor.preprocess_texts(texts)
        
        # Crear SBM
        texts_sbm = self.preprocessor.create_sbm_matrix(cleaned_texts, fit=False)
        
        # Predecir
        predictions = self.sbm_model.predict(texts_sbm)
        probabilities = self.sbm_model.predict_proba(texts_sbm)
        
        # Crear DataFrame con resultados
        results_df = pd.DataFrame({
            'texto_original': texts,
            'texto_limpio': cleaned_texts
        })
        
        # Agregar etiquetas
        for i, label in enumerate(self.label_columns):
            results_df[label] = predictions[:, i]
            results_df[f'{label}_probabilidad'] = probabilities[:, i]
        
        return results_df
    
    def format_prediction(self, prediction):
        """Formatea una predicción para mejor visualización"""
        if prediction is None:
            return "No se pudo realizar la predicción"
        
        formatted = f"📝 TEXTO ORIGINAL: {prediction['texto']}\n"
        formatted += f"🧹 TEXTO LIMPIO: {prediction['texto_limpio']}\n"
        formatted += "🏷️  ETIQUETAS PREDICHAS:\n"
        
        for i, label in enumerate(self.label_columns):
            pred_label = prediction['etiquetas_predichas'][i]
            prob = prediction['probabilidades'][i]
            status = "✅ SÍ" if pred_label == 1 else "❌ NO"
            formatted += f"   {label}: {status} (prob: {prob:.3f})\n"
        
        return formatted