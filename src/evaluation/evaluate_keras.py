import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix

# === CONFIGURACIONES ===
input_shape = (224, 224, 3)  # Mismo input que en el entrenamiento
model_path = "src/training/experiments/individual_models/resnet/checkpoints/trial_04/best_model.keras"  # Reemplaza con tu modelo .keras

# === PASO 1: CARGAR EL MODELO GUARDADO ===
model = tf.keras.models.load_model(model_path)
print("✅ Modelo cargado correctamente desde:", model_path)

# === PASO 2: PREPARAR LOS DATOS DE TEST ===
test_dir = "data/raw/Test"  # Ajusta la ruta según tu estructura

test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# === PASO 3: HACER PREDICCIONES ===
predictions = model.predict(test_generator)
threshold = 0.5  # Ajusta si es necesario
predicted_classes = (predictions > threshold).astype(int).flatten()
true_classes = test_generator.classes

# === PASO 4: CALCULAR MÉTRICAS ===
accuracy = accuracy_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes)
auc_roc = roc_auc_score(true_classes, predictions)
classification_rep = classification_report(true_classes, predicted_classes, target_names=["No Fire", "Fire"])

# === PASO 5: MOSTRAR RESULTADOS ===
print("=== Evaluating ResNet Model ===")
print(f"Model in: {model_path}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print("\nClassification Report:")
print(classification_rep)

# Matriz de confusión (opcional)
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:")
print(conf_matrix)
