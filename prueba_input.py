# Usando el predictor que creamos
from src.predictor import Predictor
# Inicializar y cargar modelos
predictor = Predictor()
if predictor.load_models():
    while True:
    # Probar con el primer relato
        texto = input("Escribe un relato (o 'salir' para terminar): ")
        if texto.lower() == "salir":
            break
        resultado1 = predictor.predict_single_text(texto)
        print(" PREDICCIÓN  Violencia física y psicológica")
        print(predictor.format_prediction(resultado1))
else:
    print("❌ No se pudieron cargar los modelos. Verifica que hayan sido entrenados correctamente.")





