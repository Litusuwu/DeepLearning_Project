import time
from tensorflow import keras

start_time = time.time()

# Cargar el modelo guardado
modelo = keras.models.load_model("/home/czegarra/proyectoDeep/DeepLearning_Project/experiments/individual_models/xception/xception_final.keras")

# Recuperar hiperparámetros generales
config = modelo.get_config()
dropout_rate = None
l2_factor = None
n_layers_to_unfreeze = 0
learning_rate = modelo.optimizer.learning_rate.numpy()

# Buscar hiperparámetros en las capas del modelo principal
for layer in modelo.layers:
    if isinstance(layer, keras.layers.Dropout):
        dropout_rate = layer.rate
    if isinstance(layer, keras.layers.Dense) and layer.kernel_regularizer:
        l2_factor = layer.kernel_regularizer.l2

# Intentar recuperar el modelo base (submodelo)
base_model = None
for layer in modelo.layers:
    if isinstance(layer, keras.Model):
        base_model = layer
        break

# Si no se encuentra el submodelo, se cuenta mediante el nombre de las capas
if base_model is None:
    print("No se encontró el modelo base como submodelo, se intentará inferir las capas del modelo base filtrando por nombre.")
    for layer in modelo.layers:
        # Suponiendo que las capas del modelo base tienen nombres que empiezan con "block"
        if layer.trainable:
            n_layers_to_unfreeze += 1
else:
    for layer in base_model.layers:
        if layer.trainable:
            n_layers_to_unfreeze += 1

end_time = time.time()

# Imprimir los hiperparámetros recuperados
print("Hiperparámetros recuperados:")
print(f"Dropout Rate: {dropout_rate}")
print(f"L2 Factor: {l2_factor}")
print(f"Número de capas desbloqueadas: {n_layers_to_unfreeze}")
print(f"Learning Rate: {learning_rate}")
print(f"Tiempo total: {(end_time - start_time):.2f} segundos")
