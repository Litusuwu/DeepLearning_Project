import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50  # Cambiado a ResNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix

# === CONFIGURACIONES ===
input_shape = (224, 224, 3)  # Mismo input que en el entrenamiento
weights_path = "kt_tuner_dir/resnet_tuning/trial_00/checkpoint.weights.h5"  # Reemplaza con tu path

# === PASO 1: RECONSTRUIR LA ARQUITECTURA DEL MODELO ===
def build_resnet_model():
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)  # SIN PESOS PRE-ENTRENADOS
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)  # Ajustar dropout si fue diferente en el entrenamiento
    predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-3))(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Construimos el modelo
model = build_resnet_model()

# Cargar los pesos entrenados
model.load_weights(weights_path)
print("✅ Pesos del modelo cargados correctamente.")

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
print(f"Model Weights in: {weights_path}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print("\nClassification Report:")
print(classification_rep)

# Matriz de confusión (opcional)
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:")
print(conf_matrix)
