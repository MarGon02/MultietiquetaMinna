# Usando el predictor que creamos
from src.predictor import Predictor

# Inicializar y cargar modelos
predictor = Predictor()
if predictor.load_models():
    
    # Probar con el primer relato
    resultado1 = predictor.predict_single_text("El niño llega al colegio con moretones en los brazos y evita hablar sobre lo sucedido. Cuando se le pregunta, baja la mirada y dice que se cayó, pero su actitud temerosa sugiere que hay más detrás de esta situación.")
    print("🔍 PREDICCIÓN 1 - Violencia física y psicológica:")
    print(predictor.format_prediction(resultado1))
    
    print("\n" + "="*60 + "\n")
    
    # Probar con el relato complejo
    resultado4 = predictor.predict_single_text("El menor presenta fractura no atendida en el brazo izquierdo y moretones en diferentes estados de cicatrización. En el colegio se muestra agresivo con sus compañeros y revela que en casa lo encierran en su habitación como castigo. Además, falta frecuentemente a clases por enfermedades no tratadas.")
    print("🔍 PREDICCIÓN 4 - Caso complejo múltiple:")
    print(predictor.format_prediction(resultado4))

else:
    print("❌ No se pudieron cargar los modelos. Verifica que hayan sido entrenados correctamente.")