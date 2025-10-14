import os
from pathlib import Path

# Configuración de paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Crear directorios si no existen
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# Configuración del modelo
MODEL_CONFIG = {
    'berto': {
        'model_name': 'dccuchile/bert-base-spanish-wwm-uncased',
        'max_length': 256,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 3
    },
    'sbm': {
        'max_features': 5000,
        'binary': True
    },
    'classifier_chain': {
        'base_estimator': 'logistic_regression',
        'max_iter': 1000,
        'random_state': 42
    }
}

# Configuración de preprocesamiento
PREPROCESSING_CONFIG = {
    'remove_special_chars': True,
    'lowercase': True,
    'remove_stopwords': True,
    'lemmatize': False  # Podemos activar después
}