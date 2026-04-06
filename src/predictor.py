import pandas as pd
import joblib
from pathlib import Path

from src.preprocessor import TextPreprocessor
from src.model_sbm import SBMClassifierChain
from src.model_berto import BETOMultiLabelClassifier
from src.config import DATA_DIR, MODELS_DIR


class Predictor:
    def __init__(self):
        self.preprocessor = TextPreprocessor()

        # Modelos
        self.sbm_model = SBMClassifierChain()
        self.berto_model = None  # se crea al cargar

        # Etiquetas
        self.label_columns = None

        # Flags
        self.is_loaded_sbm = False
        self.is_loaded_berto = False

    def load_models(
        self,
        vectorizer_file="sbm_vectorizer.joblib",
        sbm_model_file="classifier_chain_model.joblib",
        labels_file="processed_data.joblib",
        berto_dir="berto_model",
        berto_threshold=None,
    ):
        """
        Carga SBM + BETO.
        - SBM: vectorizer + classifier_chain_model.joblib
        - BETO: carpeta models/berto_model (guardada por save_model de BETO)
        """

        # 1) Cargar etiquetas reales
        try:
            processed_data = joblib.load(DATA_DIR / "processed" / labels_file)

            # Ojo: tu processed_data puede tener distintas keys según cómo lo guardaste.
            # Lo ideal es que guardes label_columns dentro.
            if isinstance(processed_data, dict) and "label_columns" in processed_data:
                self.label_columns = processed_data["label_columns"]
            else:
                # fallback: intenta inferir si viene otra estructura
                self.label_columns = processed_data.get("label_columns", None)

            if not self.label_columns:
                raise ValueError("No se encontró label_columns en processed_data.")

            print(f"✅ Etiquetas cargadas: {len(self.label_columns)} etiquetas")

        except Exception as e:
            print(f"⚠️  No se pudieron cargar etiquetas desde processed_data: {e}")
            # fallback (no recomendado para tu dataset real)
            self.label_columns = [
                "label_maltrato",
                "label_violación_del_deber_de_cuidado",
                "label_situaciones_de_riesgo",
                "label_abuso_sexual",
                "label_negligencia",
            ]

        # 2) Cargar vectorizer SBM
        if self.preprocessor.load_vectorizer(vectorizer_file):
            # 3) Cargar modelo SBM
            if self.sbm_model.load_model(sbm_model_file):
                self.is_loaded_sbm = True
                print("✅ SBM cargado correctamente")
            else:
                print("❌ No se pudo cargar el modelo SBM")
        else:
            print("❌ No se pudo cargar el vectorizer SBM")

        # 4) Cargar BETO
        try:
            self.berto_model = BETOMultiLabelClassifier(num_labels=len(self.label_columns))
            berto_path = MODELS_DIR / berto_dir

            ok = self.berto_model.load_model(str(berto_path))
            if not ok:
                print("❌ No se pudo cargar BETO desde:", berto_path)
            else:
                # aplicar umbral si lo pasas desde afuera
                if berto_threshold is not None:
                    self.berto_model.threshold = berto_threshold
                self.is_loaded_berto = True
                print("✅ BETO cargado correctamente")

        except Exception as e:
            print(f"❌ Error cargando BETO: {e}")
            self.is_loaded_berto = False

        return self.is_loaded_sbm or self.is_loaded_berto

    def predict_single_text(self, text: str):
        """
        Predice con SBM y con BETO (si están cargados).
        Devuelve un dict con ambos resultados.
        """
        if not (self.is_loaded_sbm or self.is_loaded_berto):
            print("❌ No hay modelos cargados. Ejecuta load_models() primero.")
            return None

        # Preprocesar texto (para SBM)
        cleaned_text = self.preprocessor.preprocess_texts([text])[0]

        result = {
            "texto": text,
            "texto_limpio": cleaned_text,
            "labels": self.label_columns,
            "sbm": None,
            "beto": None,
        }

        # --- SBM ---
        if self.is_loaded_sbm:
            text_sbm = self.preprocessor.create_sbm_matrix([cleaned_text], fit=False)
            pred_sbm = self.sbm_model.predict(text_sbm)[0]
            proba_sbm = self.sbm_model.predict_proba(text_sbm)[0]

            result["sbm"] = {
                "pred": pred_sbm,
                "proba": proba_sbm,
            }

        # --- BETO ---
        if self.is_loaded_berto:
            # BETO puede usar el texto original o el cleaned_text.
            # Yo recomiendo el original para no perder info, pero puedes probar.
            pred_beto, proba_beto = self.berto_model.predict([text])
            result["beto"] = {
                "pred": pred_beto[0],
                "proba": proba_beto[0],
            }

        return result

    def predict_batch(self, texts):
        """
        Predice lote con SBM y BETO, devuelve DataFrame con columnas para ambos.
        """
        if not (self.is_loaded_sbm or self.is_loaded_berto):
            print("❌ No hay modelos cargados. Ejecuta load_models() primero.")
            return None

        cleaned_texts = self.preprocessor.preprocess_texts(texts)

        df = pd.DataFrame({"texto_original": texts, "texto_limpio": cleaned_texts})

        # SBM
        if self.is_loaded_sbm:
            X_sbm = self.preprocessor.create_sbm_matrix(cleaned_texts, fit=False)
            preds = self.sbm_model.predict(X_sbm)
            probas = self.sbm_model.predict_proba(X_sbm)

            for i, label in enumerate(self.label_columns):
                df[f"SBM_{label}"] = preds[:, i]
                df[f"SBM_{label}_prob"] = probas[:, i]

        # BETO
        if self.is_loaded_berto:
            preds_b, probas_b = self.berto_model.predict(texts)

            for i, label in enumerate(self.label_columns):
                df[f"BETO_{label}"] = preds_b[:, i]
                df[f"BETO_{label}_prob"] = probas_b[:, i]

        return df

    def format_prediction(self, prediction):
        """
        Muestra resultados comparando SBM vs BETO para cada etiqueta.
        """
        if prediction is None:
            return "No se pudo realizar la predicción."

        labels = prediction["labels"]

        out = []
        out.append(f"📝 TEXTO ORIGINAL: {prediction['texto']}")
        out.append(f"🧹 TEXTO LIMPIO (SBM): {prediction['texto_limpio']}")
        out.append("")

        # Helper
        def status(x):
            return "✅ SÍ" if int(x) == 1 else "❌ NO"

        out.append("🏷️  COMPARACIÓN DE MODELOS (por etiqueta)")
        out.append("-" * 60)

        for i, label in enumerate(labels):
            line = [f"{label}:"]

            if prediction["sbm"] is not None:
                ps = prediction["sbm"]["pred"][i]
                rs = prediction["sbm"]["proba"][i]
                line.append(f"SBM={status(ps)} (p={rs:.3f})")
            else:
                line.append("SBM=—")

            if prediction["beto"] is not None:
                pb = prediction["beto"]["pred"][i]
                rb = prediction["beto"]["proba"][i]
                line.append(f"BETO={status(pb)} (p={rb:.3f})")
            else:
                line.append("BETO=—")

            out.append("   " + " | ".join(line))

        return "\n".join(out)