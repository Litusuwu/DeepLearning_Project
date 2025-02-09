import os
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# --- Funciones de utilidad ---
def load_config(config_path):
    """Carga un archivo YAML de configuración y retorna un diccionario."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    """Crea el directorio si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)

def build_xception_model(input_shape):
    """
    Construye y compila un modelo basado en Xception para clasificación binaria.
    Se utiliza la base preentrenada con pesos de ImageNet, se congela la base
    y se añade una cabeza de clasificación.
    """
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Congela la base para entrenar solo la cabeza

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    # Para clasificación binaria, usamos 1 unidad y activación sigmoide
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# --- Script Principal ---
if __name__ == '__main__':
    # Cargar configuraciones desde YAML
    train_config = load_config("config/train_config.yaml")
    data_config = load_config("config/data_paths.yaml")
    
    # Parámetros de entrenamiento (con valores por defecto si no se especifican)
    epochs = train_config.get("epochs", 10)
    batch_size = train_config.get("batch_size", 32)
    input_shape = tuple(train_config.get("input_shape", [224, 224, 3]))
    optimizer = train_config.get("optimizer", "adam")
    loss = train_config.get("loss", "binary_crossentropy")
    metrics = train_config.get("metrics", ["accuracy"])
    
    # Directorios de salida (logs, checkpoints, results) para Xception
    output_dirs = train_config.get("output_dirs", {})
    logs_dir = output_dirs.get("logs", "experiments/individual_models/xception/logs")
    checkpoints_dir = output_dirs.get("checkpoints", "experiments/individual_models/xception/checkpoints")
    results_dir = output_dirs.get("results", "experiments/individual_models/xception/results")
    
    ensure_dir(logs_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(results_dir)
    
    # Rutas de datos: se asume que los datos ya están organizados en carpetas
    train_dir = data_config.get("train_dir", "data/Training")
    val_dir   = data_config.get("val_dir", None)  # Si tienes una carpeta de validación separada
    
    # Crear generadores de datos
    # Para clasificación binaria usamos 'binary' como class_mode
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=train_config.get("augmentation", {}).get("shear_range", 0.0),
        zoom_range=train_config.get("augmentation", {}).get("zoom_range", 0.0),
        horizontal_flip=train_config.get("augmentation", {}).get("horizontal_flip", False),
        validation_split=train_config.get("validation_split", 0.0)
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Si no se tiene carpeta de validación separada, usar validation_split
    if val_dir is None and train_config.get("validation_split", 0) > 0:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=input_shape[:2],
            batch_size=batch_size,
            class_mode='binary',
            subset='training'
        )
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=input_shape[:2],
            batch_size=batch_size,
            class_mode='binary',
            subset='validation'
        )
    else:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=input_shape[:2],
            batch_size=batch_size,
            class_mode='binary'
        )
        validation_generator = None
        if val_dir is not None:
            validation_generator = test_datagen.flow_from_directory(
                val_dir,
                target_size=input_shape[:2],
                batch_size=batch_size,
                class_mode='binary',
                shuffle=False
            )
    
    # Construir el modelo Xception
    model = build_xception_model(input_shape=input_shape)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # Configurar callbacks
    checkpoint_path = os.path.join(checkpoints_dir, "xception_best.h5")
    csv_logger_path = os.path.join(logs_dir, "xception_training.csv")
    
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 monitor=train_config.get("early_stopping", {}).get("monitor", "val_loss"),
                                 save_best_only=True,
                                 verbose=1)
    
    early_stopping = EarlyStopping(monitor=train_config.get("early_stopping", {}).get("monitor", "val_loss"),
                                   patience=train_config.get("early_stopping", {}).get("patience", 3),
                                   verbose=1)
    
    csv_logger = CSVLogger(csv_logger_path)
    
    callbacks_list = [checkpoint, early_stopping, csv_logger]
    
    # Entrenar el modelo
    if validation_generator is not None:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=epochs,
            callbacks=callbacks_list
        )
    else:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            callbacks=callbacks_list
        )
    
    # Guardar el modelo final en la carpeta de resultados
    final_model_path = os.path.join(results_dir, "xception_final.h5")
    model.save(final_model_path)
    print("Modelo Xception final guardado en:", final_model_path)
