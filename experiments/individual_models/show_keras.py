from tensorflow import keras

# Ruta completa al modelo en la carpeta
model_path = "/home/czegarra/proyectoDeep/DeepLearning_Project/experiments/individual_models/xception/xception_final.keras"
# Cargar el modelo
model = keras.models.load_model(model_path)

# Mostrar el resumen del modelo para ver qué arquitectura tiene
model.summary()
