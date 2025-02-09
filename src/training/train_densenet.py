import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import os

# Define las rutas a tus carpetas de entrenamiento y prueba.
# Asegúrate de que en "Training" y "Test" existan dos subcarpetas (por ejemplo, "fire" y "nofire")
train_dir = "/home/litus/Documents/Universidad/DeepLearning/Training/Training"
test_dir  = "/home/litus/Documents/Universidad/DeepLearning/Test/Test"

# Parámetros
batch_size = 32
img_height = 224  # DenseNet121 utiliza 224x224 por defecto
img_width = 224

# Configura el generador para entrenamiento con data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Para el conjunto de prueba se aplica solo el rescale
test_datagen = ImageDataGenerator(rescale=1./255)

# Crea el generador para el conjunto de entrenamiento (usamos 'binary' para etiquetas binarias)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Crea el generador para el conjunto de prueba
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Carga el modelo DenseNet121 preentrenado en ImageNet, sin la capa superior
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Congela las capas del modelo base para entrenar inicialmente solo la cabeza
base_model.trainable = False

# Agrega la cabeza de clasificación
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Ayuda a prevenir el sobreajuste
# Para clasificación binaria, usamos 1 salida con activación sigmoide
predictions = Dense(1, activation='sigmoid')(x)

# Define el modelo completo
model = Model(inputs=base_model.input, outputs=predictions)

# Compila el modelo con binary_crossentropy
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Muestra un resumen del modelo
model.summary()
print("Número de clases en Training:", train_generator.num_classes)

# Entrena el modelo
epochs = 10  # Puedes ajustar el número de épocas según sea necesario
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    epochs=epochs
)

# Guarda el modelo entrenado (opcional)
model.save("modelo_densenet.h5")
