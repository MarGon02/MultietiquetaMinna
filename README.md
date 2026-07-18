# MultietiquetaMinna

Guía de uso de los scripts del proyecto: cómo instalar, entrenar, evaluar y probar los modelos.

> ⚠️ Los comandos se muestran en su forma estándar (`python archivo.py`). Si algún script pide argumentos específicos no confirmados acá, conviene revisar el código antes de correrlo.

## Estructura del repositorio

```
MultietiquetaMinna/
├── data/
│   └── raw/                  # Datos crudos (relatos anonimizados)
├── src/
│   ├── config.py              # Configuración general (hiperparámetros, rutas, etc.)
│   ├── data_loader.py         # Carga y lectura del dataset
│   ├── preprocessor.py        # Preprocesamiento de texto (limpieza, stopwords, etc.)
│   ├── model_sbm.py            # Definición del modelo SBM + Classifier Chain
│   ├── model_berto.py          # Definición del modelo BETO (fine-tuning)
│   ├── trainer.py              # Lógica de entrenamiento de los modelos
│   ├── threshold_tuning.py     # Optimización de umbrales de decisión por etiqueta
│   ├── evaluator.py            # Cálculo de métricas (F1 Macro/Micro, Precision, Recall, etc.)
│   ├── kfold_validation.py     # Lógica de validación cruzada K-Fold
│   ├── predictor.py            # Lógica de predicción sobre texto nuevo
│   └── utils.py                # Funciones auxiliares
├── app.py                     # Interfaz / API de demo (FastAPI)
├── main.py                    # Script principal de entrenamiento (el que se usa realmente)
├── main_2.py                  # Versión anterior/alternativa de entrenamiento (no es la vigente)
├── comparar_modelos.py         # Comparación de resultados entre SBM y BETO
├── kfold_analysis.py           # Análisis K-Fold (solo validación cruzada)
├── kfold_analysis_v2.py        # Análisis K-Fold + evaluación contra el test final
├── prueba.py                   # Script de prueba/testeo
├── prueba_input.py             # Ingresa un texto y devuelve la predicción del modelo
└── requirements.txt            # Dependencias del proyecto
```

## Instalación

```bash
pip install -r requirements.txt
```

Entorno recomendado: Google Colab (con GPU) para el entrenamiento de BETO, dado el costo computacional del fine-tuning.

## Entrenamiento de los modelos

`main.py` es el script vigente para entrenar ambos modelos (SBM y BETO). `main_2.py` es una versión anterior y no debería usarse como referencia actual.

```bash
python main.py
```

Este script se apoya en los módulos de `src/`: `data_loader.py` para cargar el dataset, `preprocessor.py` para el preprocesamiento (distinto para SBM y para BETO), `model_sbm.py` / `model_berto.py` para la definición de cada modelo, `trainer.py` para el entrenamiento propiamente dicho, y `threshold_tuning.py` para la calibración de umbrales por etiqueta.

## Análisis K-Fold

Hay dos scripts distintos según qué querés analizar:

- **`kfold_analysis.py`** → corre únicamente la validación cruzada K-Fold (K=5) sobre los datos de desarrollo.
- **`kfold_analysis_v2.py`** → además del K-Fold, evalúa/calibra contra el test final (los 60 relatos nunca vistos).

```bash
python kfold_analysis.py
# o
python kfold_analysis_v2.py
```

Ambos se apoyan en `src/kfold_validation.py` para la lógica de partición y repetición de folds.

## Comparación de modelos

`comparar_modelos.py` genera la comparación de métricas entre SBM y BETO (los resultados que después van a las tablas del informe/tesis).

```bash
python comparar_modelos.py
```

## Predicción sobre un texto nuevo

`prueba_input.py` pide el texto de forma **interactiva** en consola (usa `input()`), así que hay que correrlo directo en terminal — no funciona bien pegado en una celda de notebook/Colab:

```bash
python prueba_input.py
```

Internamente usa la clase `Predictor` (`src/predictor.py`): `load_models()` para cargar los `.joblib` desde `MODELS_DIR` (definido en `config.py` como `BASE_DIR / "models"`), `predict_single_text()` para predecir, y `format_prediction()` para mostrar el resultado formateado en consola.

> Nota: en una versión de `predictor.py` que vimos en una sesión anterior, la parte de BETO estaba comentada y `format_prediction()` mostraba solo los resultados de SBM. No tengo forma de confirmar si esto sigue así en la versión actual — convendría que lo revises en el archivo.

## Interfaz / API de demo

`app.py` levanta un servidor FastAPI que expone la misma lógica de `Predictor`. El HTML de la interfaz (`templates/index.html`) se lee directamente desde disco con `pathlib` y se devuelve como `HTMLResponse` — **no usa Jinja2**, porque hubo un bug de compatibilidad (`TypeError: unhashable type: 'dict'`) entre versiones recientes de Starlette/FastAPI y Jinja2.

Ejecutar parado en la raíz del proyecto:

```bash
python -m uvicorn app:app --reload
```

(`python app.py` puede fallar si `uvicorn` no está reconocido como comando directo en la terminal — por eso se usa `python -m uvicorn`.)

Luego abrir **http://localhost:8000** en el navegador. Endpoints:
- `GET /` → sirve la interfaz HTML
- `POST /clasificar` → recibe `{"texto": "..."}` y devuelve la predicción en JSON

## Datos

`data/raw/` contiene el dataset crudo que usa `data_loader.py`.

## `prueba.py`

Script de testeo/debug, no forma parte del flujo principal (entrenamiento → evaluación → demo).
