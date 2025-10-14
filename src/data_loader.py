import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from .config import DATA_DIR, MODELS_DIR

class DataLoader:
    def __init__(self, data_path=None):
        self.data_path = data_path or DATA_DIR / "raw" / "relatos_minna.csv"
        
    def load_data(self):
        """Carga los datos desde el archivo CSV"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Datos cargados: {self.data.shape[0]} filas, {self.data.shape[1]} columnas")
            return self.data
        except FileNotFoundError:
            print(f"❌ Archivo no encontrado: {self.data_path}")
            return None
    
    def explore_data(self):
        """Exploración básica de los datos"""
        if not hasattr(self, 'data'):
            self.load_data()
            
        print("=== INFORMACIÓN DEL DATASET ===")
        print(self.data.info())
        
        print("\n=== PRIMERAS FILAS ===")
        print(self.data.head())
        
        print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
        print(self.data.describe())
        
        # Identificar columnas de texto y etiquetas
        self.text_column = 'relato'  # Ajustar según tu dataset
        self.label_columns = [col for col in self.data.columns 
                            if col not in [self.text_column, 'id']]
        
        print(f"\n=== COLUMNA DE TEXTO: {self.text_column} ===")
        print(f"=== COLUMNAS DE ETIQUETAS: {self.label_columns} ===")
        
        return self.text_column, self.label_columns
    
    def split_data(self, test_size=0.2, random_state=42):
        """Divide los datos en entrenamiento y prueba"""
        if not hasattr(self, 'data'):
            self.load_data()
            
        X = self.data[self.text_column]
        y = self.data[self.label_columns]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✅ Datos divididos:")
        print(f"   Entrenamiento: {X_train.shape[0]} ejemplos")
        print(f"   Prueba: {X_test.shape[0]} ejemplos")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, filename='processed_data.joblib'):
        """Guarda los datos procesados"""
        processed_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label_columns': self.label_columns
        }
        
        save_path = DATA_DIR / "processed" / filename
        save_path.parent.mkdir(exist_ok=True)
        
        joblib.dump(processed_data, save_path)
        print(f"✅ Datos procesados guardados en: {save_path}")
    
    def load_processed_data(self, filename='processed_data.joblib'):
        """Carga los datos procesados"""
        load_path = DATA_DIR / "processed" / filename
        
        try:
            processed_data = joblib.load(load_path)
            print(f"✅ Datos procesados cargados desde: {load_path}")
            return processed_data
        except FileNotFoundError:
            print(f"❌ Archivo no encontrado: {load_path}")
            return None