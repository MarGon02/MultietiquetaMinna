import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from pathlib import Path
from src.config import DATA_DIR, MODELS_DIR, PREPROCESSING_CONFIG

# Descargar recursos de NLTK si es necesario
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self, config=PREPROCESSING_CONFIG):
        self.config = config
        self.stopwords_es = set(stopwords.words('spanish'))
        self.vectorizer = None
    
    def clean_text(self, text):
        """Limpia y normaliza un texto individual"""
        if pd.isna(text):
            return ""
        
        # Convertir a minúsculas
        if self.config['lowercase']:
            text = text.lower()
        
        # Eliminar caracteres especiales
        if self.config['remove_special_chars']:
            text = re.sub(r'[^a-záéíóúñü\s]', '', text)
        
        # Eliminar espacios extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """Elimina palabras vacías"""
        if self.config['remove_stopwords']:
            words = text.split()
            filtered_words = [word for word in words if word not in self.stopwords_es]
            return ' '.join(filtered_words)
        return text
    
    def preprocess_texts(self, texts):
        """Preprocesa una lista de textos"""
        print("🔄 Preprocesando textos...")
        
        # Aplicar limpieza
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Aplicar eliminación de stopwords
        if self.config['remove_stopwords']:
            cleaned_texts = [self.remove_stopwords(text) for text in cleaned_texts]
        
        print("✅ Preprocesamiento completado")
        return cleaned_texts
    
    def create_sbm_matrix(self, texts, fit=True):
        """Crea la Matriz Binaria Dispersa"""
        if self.vectorizer is None or fit:
            self.vectorizer = CountVectorizer(
                binary=self.config.get('binary', True),
                max_features=self.config.get('max_features', 5000),
                stop_words=list(self.stopwords_es) if self.config['remove_stopwords'] else None
            )
        
        if fit:
            X_sbm = self.vectorizer.fit_transform(texts)
            print(f"✅ SBM creado (entrenamiento): {X_sbm.shape}")
        else:
            X_sbm = self.vectorizer.transform(texts)
            print(f"✅ SBM creado (transformación): {X_sbm.shape}")
        
        return X_sbm
    
    def save_vectorizer(self, filename='sbm_vectorizer.joblib'):
        """Guarda el vectorizer entrenado"""
        if self.vectorizer is not None:
            save_path = MODELS_DIR / filename
            joblib.dump(self.vectorizer, save_path)
            print(f"✅ Vectorizer guardado en: {save_path}")
    
    def load_vectorizer(self, filename='sbm_vectorizer.joblib'):
        """Carga el vectorizer entrenado"""
        load_path = MODELS_DIR / filename
        try:
            self.vectorizer = joblib.load(load_path)
            print(f"✅ Vectorizer cargado desde: {load_path}")
            return True
        except FileNotFoundError:
            print(f"❌ Vectorizer no encontrado: {load_path}")
            return False