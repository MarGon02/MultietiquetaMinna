#!/usr/bin/env python3
"""
Script para probar y comparar AMBOS modelos (SBM y BETO)
"""

import sys
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from src.predictor import Predictor
from src.model_berto import BETOMultiLabelClassifier
from src.data_loader import DataLoader
import joblib

class ComparadorModelos:
    def __init__(self):
        self.predictor_sbm = Predictor()
        self.modelo_berto = None
        self.label_columns = None
        
    def cargar_modelos(self):
        """Carga ambos modelos entrenados"""
        print("🔄 Cargando modelos entrenados...")
        
        # Cargar modelo SBM
        print("📋 Cargando SBM + Classifier Chain...")
        if not self.predictor_sbm.load_models():
            print("❌ No se pudo cargar el modelo SBM")
            return False
        
        # Cargar modelo BETO
        print("🤖 Cargando BETO...")
        try:
            # Cargar etiquetas para saber la cantidad
            loader = DataLoader()
            data = loader.load_data()
            if data is not None:
                text_col, label_cols = loader.explore_data()
                self.label_columns = label_cols
                
                # Inicializar y cargar BETO
                self.modelo_berto = BETOMultiLabelClassifier(num_labels=len(label_cols))
                if self.modelo_berto.load_model("models/berto_model"):
                    print("✅ BETO cargado correctamente")
                else:
                    print("⚠️  No se pudo cargar BETO. ¿Lo entrenaste?")
                    self.modelo_berto = None
            else:
                print("⚠️  No se pudieron cargar las etiquetas, usando predeterminadas")
                self.label_columns = ['violencia_fisica', 'violencia_psicologica', 'abandono', 'abuso_sexual', 'negligencia']
                self.modelo_berto = None
                
        except Exception as e:
            print(f"⚠️  Error cargando BETO: {e}")
            self.modelo_berto = None
        
        return True
    
    def predecir_con_ambos(self, texto):
        """Realiza predicciones con ambos modelos"""
        resultados = {}
        
        # Predicción con SBM
        print("🔍 PREDICIENDO CON SBM...")
        resultado_sbm = self.predictor_sbm.predict_single_text(texto)
        resultados['SBM'] = resultado_sbm
        
        # Predicción con BETO (si está disponible)
        if self.modelo_berto:
            print("🔍 PREDICIENDO CON BETO...")
            try:
                y_pred_berto, y_proba_berto = self.modelo_berto.predict([texto])
                
                resultado_berto = {
                    'texto': texto,
                    'texto_limpio': texto,  # BETO hace su propio preprocesamiento
                    'etiquetas_predichas': y_pred_berto[0],
                    'probabilidades': y_proba_berto[0]
                }
                resultados['BETO'] = resultado_berto
            except Exception as e:
                print(f"❌ Error en predicción BETO: {e}")
                resultados['BETO'] = None
        else:
            resultados['BETO'] = None
            
        return resultados
    
    def formatear_comparacion(self, resultados, descripcion=""):
        """Formatea la comparación entre ambos modelos"""
        if not resultados:
            return "No se pudieron obtener resultados"
        
        comparacion = f"\n🎯 {descripcion}\n" if descripcion else "\n🎯 RESULTADOS:\n"
        comparacion += "=" * 80 + "\n"
        
        # Resultados SBM
        if 'SBM' in resultados and resultados['SBM']:
            comparacion += "\n📊 MODELO SBM + CLASSIFIER CHAIN:\n"
            comparacion += "-" * 40 + "\n"
            comparacion += self.predictor_sbm.format_prediction(resultados['SBM'])
        
        # Resultados BETO
        if 'BETO' in resultados and resultados['BETO']:
            comparacion += "\n📊 MODELO BETO (TRANSFORMERS):\n"
            comparacion += "-" * 40 + "\n"
            
            pred = resultados['BETO']
            comparacion += f"📝 TEXTO: {pred['texto'][:100]}...\n"
            comparacion += "🏷️  ETIQUETAS PREDICHAS:\n"
            
            for i, label in enumerate(self.label_columns):
                pred_label = pred['etiquetas_predichas'][i]
                prob = pred['probabilidades'][i]
                status = "✅ SÍ" if pred_label == 1 else "❌ NO"
                comparacion += f"   {label}: {status} (prob: {prob:.3f})\n"
        
        elif resultados['BETO'] is None:
            comparacion += "\n📊 MODELO BETO: ❌ No disponible\n"
            comparacion += "💡 Ejecuta main.py primero para entrenar BETO\n"
        
        comparacion += "=" * 80
        return comparacion

# USO DEL COMPARADOR
def main():
    # Textos de prueba
    texto1 = "El niño llega al colegio con moretones en los brazos y evita hablar sobre lo sucedido. Cuando se le pregunta, baja la mirada y dice que se cayó, pero su actitud temerosa sugiere que hay más detrás de esta situación."
    
    texto4 = "El menor presenta fractura no atendida en el brazo izquierdo y moretones en diferentes estados de cicatrización. En el colegio se muestra agresivo con sus compañeros y revela que en casa lo encierran en su habitación como castigo. Además, falta frecuentemente a clases por enfermedades no tratadas."
    
    # Inicializar comparador
    comparador = ComparadorModelos()
    
    if comparador.cargar_modelos():
        print("\n" + "="*80)
        print("🤖 COMPARACIÓN DE MODELOS PLN - MINNA")
        print("="*80)
        
        # Probar con el primer relato
        print("\n1. PROBANDO RELATO - Violencia física y psicológica:")
        resultados1 = comparador.predecir_con_ambos(texto1)
        print(comparador.formatear_comparacion(resultados1, "VIOLENCIA FÍSICA Y PSICOLÓGICA"))
        
        print("\n" + "="*80)
        
        # Probar con el relato complejo
        print("\n2. PROBANDO RELATO - Caso complejo múltiple:")
        resultados4 = comparador.predecir_con_ambos(texto4)
        print(comparador.formatear_comparacion(resultados4, "CASO COMPLEJO MÚLTIPLE"))
        
        print("\n" + "="*80)
        print("🎉 COMPARACIÓN COMPLETADA")
        
    else:
        print("❌ No se pudieron cargar los modelos. Verifica que:")
        print("   - Hayas ejecutado: python main.py")
        print("   - Los modelos estén en la carpeta 'models/'")

if __name__ == "__main__":
    main()