import os
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# Habilitar optimizaciones para la GPU
tf.config.optimizer.set_jit(True)  # Habilitar XLA (Accelerated Linear Algebra)
tf.keras.mixed_precision.set_global_policy('mixed_float16')  # Usar mixed precision

def load_config(config_path):
    """Carga un archivo YAML de configuración."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    """Crea un directorio si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)

def build_model(hp, input_shape, workers=1, use_multiprocessing=False):
    """Construye el modelo Xception con hiperparámetros a optimizar."""
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.05, default=0.3)
    l2_factor = hp.Choice('l2_factor', values=[1e-4, 5e-4, 1e-3], default=1e-4)
    n_layers_to_unfreeze = hp.Int('n_layers_to_unfreeze', min_value=5, max_value=30, step=5, default=10)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-4)

    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Congela la base inicialmente

    # Descongelar las últimas n capas del modelo para fine-tuning
    for layer in base_model.layers[-n_layers_to_unfreeze:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_factor))(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Guardar workers y multiprocessing como atributos del modelo
    model.workers = workers
    model.use_multiprocessing = use_multiprocessing

    return model

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_config = load_config("config/train_config_xception.yaml")
    data_config = load_config("config/data_paths.yaml")

    num_workers = train_config.get('num_workers', 1)
    use_multiprocessing = train_config.get('use_multiprocessing', False)
    epochs = train_config.get("epochs", 20)
    batch_size = train_config.get("batch_size", 64)
    input_shape = tuple(train_config.get("input_shape", [224, 224, 3]))

    # Crear directorios para experimentos
    experiments_dir = os.path.abspath(os.path.join(script_dir, "experiments/individual_models/xception_og"))
    logs_dir = os.path.join(experiments_dir, "logs")
    ensure_dir(logs_dir)

    train_dir = data_config.get("train_dir", "data/processed/Training")
    validation_split = train_config.get("validation_split", 0.2)

    # Configuración de Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=train_config.get("augmentation", {}).get("shear_range", 0.1),
        zoom_range=train_config.get("augmentation", {}).get("zoom_range", 0.2),
        horizontal_flip=train_config.get("augmentation", {}).get("horizontal_flip", True),
        rotation_range=train_config.get("augmentation", {}).get("rotation_range", 15),
        brightness_range=train_config.get("augmentation", {}).get("brightness_range", [0.8, 1.2]),
        validation_split=validation_split
    )

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
    ) if validation_split > 0 else None

    # Configuración del Keras Tuner con workers y use_multiprocessing
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(hp, input_shape, workers=num_workers, use_multiprocessing=use_multiprocessing),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=2,
        directory='kt_tuner_dir',
        project_name='xception_tuning_og'
    )

    # Callback para guardar el mejor modelo
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(experiments_dir, "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        mode='min',
        save_weights_only=False
    )

    # Entrenar con Keras Tuner
    tuner.search(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[
            checkpoint_callback,
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=train_config.get("early_stopping", {}).get("patience", 5)
            )
        ]
    )

    # Obtener los mejores hiperparámetros
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Mejores hiperparámetros encontrados para Xception:")
    print(f"  - Dropout Rate: {best_hp.get('dropout_rate')}")
    print(f"  - L2 Factor: {best_hp.get('l2_factor')}")
    print(f"  - Número de capas a descongelar: {best_hp.get('n_layers_to_unfreeze')}")
    print(f"  - Learning Rate: {best_hp.get('learning_rate')}")

    # Construir y guardar el mejor modelo
    best_model = tuner.hypermodel.build(best_hp)
    model_save_path = os.path.join("kt_tuner_dir/xception_tuning_og", "final_best_model.keras")
    best_model.save(model_save_path, save_format="keras")
    print(f"Modelo guardado: {model_save_path}") 