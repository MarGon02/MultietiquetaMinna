# Usando el predictor que creamos
from src.predictor import Predictor

# Inicializar y cargar modelos
predictor = Predictor()
if predictor.load_models():
    
    # Probar con el primer relato
    resultado1 = predictor.predict_single_text("La recurrente solicitó confidencialidad, reporta la situación de maltrato que reciben sus sobrinos por parte de la madre:   Se trata del adolescente NNA de 16 años; NNA de 11 años; NNA de 8 años y del niño NNA de 5 años, hijos de los Sres. SUJETO de 31 años y SUJETO de 42 años, los padres están separados, la madre y los NNAs viven frente a su domicilio, refiere que el día de hoy la niña NNA de 11 años fue a su domicilio pidiendo comida, su padre telefónicamente le habría dicho que vuelva a su casa y pida a su madre les prepare la comida, la niña se negó al pedido del padre de forma enfática alegando que su madre la maltratará si le hace eso.   Refiere la recurrente que la madre de los niños no los cuida como corresponde, que los niños llegan  a las15:00 hs de la escuela y no encuentran comida, le envía a la escuela sin merienda.  por lo que acostumbran a acudir a la casa de los abuelos para alimentarse. Tambien refiere que la madre maltrata sistemáticamente a los niños de forma verbal y física. La Sra. SUJETO se dirige a sus hijos con la sgtes palabras: ""cara de mongo"", ""No servís para  nada"", ""Les hare pasar lo que yo vivé en mi niñez"". Refiere tambien que la madre de los niños es una persona emocionalmente inestable, necesita asistencia psicológica y/o Tto psiquiátrico,  cuando pega a su hijos lo hace sin compasión, muy mal les pega, anteriormente tambien agredía al padre de los niños hace un año atrás había pateado al Sr. SUJETO estando uno de sus hijos en brazos del Sr. SUJETO.-   El padre de los niños es hermano de la recurrente y le prohibió que denuncie los hechos, el Sr. SUJETO tendría miedo de que sus hijos sean llevados a un albergue, por tal motivo no denuncia los hechos de maltrato que reciben.-   La recurrente reiteró su pedido de confidencialidad, le mencionó a la Fiscal SUJETO, que la misma le orientó que llame al servicio a reportar el caso de forma confidencial.-  ")
    print("🔍 PREDICCIÓN 1 - Violencia física y psicológica:")
    print(predictor.format_prediction(resultado1))
    
    print("\n" + "="*60 + "\n")
    
    # Probar con el relato complejo
    resultado4 = predictor.predict_single_text("El menor presenta fractura no atendida en el brazo izquierdo y moretones en diferentes estados de cicatrización. En el colegio se muestra agresivo con sus compañeros y revela que en casa lo encierran en su habitación como castigo. Además, falta frecuentemente a clases por enfermedades no tratadas.")
    print("🔍 PREDICCIÓN 4 - Caso complejo múltiple:")
    print(predictor.format_prediction(resultado4))

else:
    print("❌ No se pudieron cargar los modelos. Verifica que hayan sido entrenados correctamente.")