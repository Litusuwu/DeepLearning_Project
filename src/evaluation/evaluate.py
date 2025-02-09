import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ruta donde se encuentran las imágenes de test.
test_dir = "/home/litus/Documents/Universidad/DeepLearning/Test"

# Parámetros (ajusta si es necesario)
img_height = 224
img_width = 224
batch_size = 32

# Crea el generador para el conjunto de test.
# Es importante usar shuffle=False para que el orden de las predicciones
# coincida con el de las etiquetas verdaderas.
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",  # O "categorical" si usaste 2 neuronas en la salida
    shuffle=False
)

# Carga el modelo entrenado.
model = tf.keras.models.load_model("models/modelo_densenet.h5")

# Obtén las predicciones para todo el conjunto de test.
# La función predict devuelve una matriz; en el caso binario, la salida será de forma (N,1).
predictions = model.predict(test_generator, steps = np.ceil(test_generator.samples / batch_size))

# Convierte las predicciones en etiquetas:
# Para clasificación binaria, se usa un umbral de 0.5.
predicted_classes = (predictions > 0.5).astype("int32").flatten()

# Obtén las etiquetas verdaderas del generador.
true_classes = test_generator.classes

# Calcula las métricas de performance.
accuracy = accuracy_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes)
report = classification_report(true_classes, predicted_classes, target_names=list(test_generator.class_indices.keys()))

print("=== Performance on Test Data ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(report)
