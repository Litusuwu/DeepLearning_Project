import os
import json
import yaml
import tensorflow as tf
import keras_tuner as kt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# IMPORTAR ResNet101
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def load_config(config_path):
    """Carga un archivo YAML y lo retorna como un dict."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    """Crea el directorio y subdirectorios si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)

def build_model(hp, input_shape):
    """
    Construye el modelo ResNet101 con hiperparámetros ajustables:
    - dropout_rate (0.2 a 0.5 con step 0.05)
    - l2_factor (1e-4, 5e-4 o 1e-3)
    - n_layers_to_unfreeze (5, 10, 15, 20)
    - learning_rate (1e-4 a 1e-2 en log scale)
    """
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.05, default=0.3)
    l2_factor = hp.Choice('l2_factor', values=[1e-4, 5e-4, 1e-3], default=1e-4)
    n_layers_to_unfreeze = hp.Int('n_layers_to_unfreeze', min_value=5, max_value=20, step=5, default=10)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-4)
    
    # USAR ResNet101 en lugar de ResNet50
    base_model = ResNet101(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    # Descongelar las últimas n capas
    for layer in base_model.layers[-n_layers_to_unfreeze:]:
        layer.trainable = True
    
    # Construir la parte final
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_factor))(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

class CustomTuner(kt.RandomSearch):
    """
    Tuner personalizado que guarda, al final de cada trial:
    - build_config.json con los hiperparámetros
    - best_model.keras con la arquitectura y pesos
    - checkpoint.weights.h5 solo los pesos
    - trial.json con información general del trial
    """
    def on_trial_end(self, trial):
        trial_id = trial.trial_id
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        experiments_dir = os.path.abspath(os.path.join(script_dir, "experiments/individual_models/resnet"))
        trial_dir = os.path.join(experiments_dir, "checkpoints", f"trial_{trial_id}")
        ensure_dir(trial_dir)
        
        # Guardar hiperparámetros del trial
        hp_config = trial.hyperparameters.values
        with open(os.path.join(trial_dir, "build_config.json"), "w") as f:
            json.dump(hp_config, f, indent=4)
        
        # Reconstruir y compilar con estos hiperparámetros
        best_model = self.hypermodel.build(trial.hyperparameters)
        best_model.compile(
            optimizer=Adam(learning_rate=trial.hyperparameters.get('learning_rate')),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Guardar el modelo completo
        model_save_path = os.path.join(trial_dir, "best_model.keras")
        best_model.save(model_save_path)
        print(f"✅ Modelo guardado para Trial {trial_id}: {model_save_path}")
        
        # Guardar los pesos
        weights_save_path = os.path.join(trial_dir, "checkpoint.weights.h5")
        best_model.save_weights(weights_save_path)
        print(f"✅ Pesos guardados para Trial {trial_id}: {weights_save_path}")
        
        # Guardar info del trial
        trial_info = {
            "trial_id": trial_id,
            "score": trial.score if trial.score is not None else "N/A",
            "best_hyperparameters": hp_config
        }
        with open(os.path.join(trial_dir, "trial.json"), "w") as f:
            json.dump(trial_info, f, indent=4)
        
        print(f"📊 Información guardada para Trial {trial_id}")
        # Llamar al método base
        super().on_trial_end(trial)

if __name__ == '__main__':
    # Directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Cargar las configuraciones
    train_config = load_config("config/train_config_resnet.yaml")
    data_config = load_config("config/data_paths.yaml")
    
    epochs = train_config.get("epochs", 10)
    batch_size = train_config.get("batch_size", 8)
    input_shape = tuple(train_config.get("input_shape", [224, 224, 3]))
    train_dir = data_config.get("train_dir", "data/processed/Training")
    validation_split = train_config.get("validation_split", 0.0)

    experiments_dir = os.path.abspath(os.path.join(script_dir, "experiments/individual_models/resnet"))
    logs_dir = os.path.join(experiments_dir, "logs")
    ensure_dir(logs_dir)

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=train_config.get("augmentation", {}).get("shear_range", 0.0),
        zoom_range=train_config.get("augmentation", {}).get("zoom_range", 0.0),
        horizontal_flip=train_config.get("augmentation", {}).get("horizontal_flip", False),
        rotation_range=10,
        brightness_range=[0.8, 1.2],
        validation_split=validation_split
    )

    if validation_split > 0:
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

    # Instanciar el tuner personalizado
    tuner = CustomTuner(
        hypermodel=lambda hp: build_model(hp, input_shape),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='kt_tuner_dir',
        project_name='resnet_tuning'
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Realizar la búsqueda de hiperparámetros
    if validation_generator is not None:
        tuner.search(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[early_stop]
        )
    else:
        tuner.search(
            train_generator,
            epochs=epochs,
            callbacks=[early_stop]
        )
    
    print("✅ Proceso de tuning finalizado. Modelos guardados por trial en 'experiments/individual_models/resnet/checkpoints/'")
