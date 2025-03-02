import os
import json
import yaml
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def build_model(hp, input_shape):
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.05, default=0.3)
    l2_factor = hp.Choice('l2_factor', values=[1e-4, 5e-4, 1e-3], default=1e-4)
    n_layers_to_unfreeze = hp.Int('n_layers_to_unfreeze', min_value=5, max_value=50, step=5, default=10)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-4)
    
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    for layer in base_model.layers[-n_layers_to_unfreeze:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_factor))(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

class TrialModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, trial_id, base_dir):
        super(TrialModelCheckpoint, self).__init__()
        self.trial_id = trial_id
        self.base_dir = base_dir
        self.best_val_loss = float('inf')
        self.trial_dir = os.path.join(self.base_dir, f"trial_{self.trial_id}")
        ensure_dir(self.trial_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss")
        if val_loss and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            model_save_path = os.path.join(self.trial_dir, "best_model_xception.keras")
            self.model.save(model_save_path)
            print(f"Trial {self.trial_id}: Mejor modelo guardado con val_loss {val_loss:.4f} en epoch {epoch}")

class CustomTuner(kt.RandomSearch):
    def on_trial_end(self, trial):
        trial_id = trial.trial_id
        script_dir = os.path.dirname(os.path.abspath(__file__))
        experiments_dir = os.path.abspath(os.path.join(script_dir, "experiments/individual_models/xception3"))
        trial_dir = os.path.join(experiments_dir, "checkpoints", f"trial_{trial_id}")
        ensure_dir(trial_dir)

        hp_config = trial.hyperparameters.values
        with open(os.path.join(trial_dir, "build_config.json"), "w") as f:
            json.dump(hp_config, f, indent=4)

        best_model_path = os.path.join(trial_dir, "best_model_xception.keras")
        if os.path.exists(best_model_path):
            best_model = tf.keras.models.load_model(best_model_path)
        else:
            print(f"No se encontró un modelo guardado en {best_model_path}, guardando la última versión entrenada.")
            best_model = self.hypermodel.build(trial.hyperparameters)
            best_model.compile(
                optimizer=Adam(learning_rate=trial.hyperparameters.get('learning_rate')),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            best_model.save(best_model_path)

        weights_save_path = os.path.join(trial_dir, "checkpoint.weights.h5")
        best_model.save_weights(weights_save_path)

        trial_info = {
            "trial_id": trial_id,
            "score": trial.score if trial.score is not None else "N/A",
            "best_hyperparameters": hp_config
        }
        with open(os.path.join(trial_dir, "trial.json"), "w") as f:
            json.dump(trial_info, f, indent=4)

        print(f"📊 Trial {trial_id}: Información y modelo guardados correctamente.")
        
        super().on_trial_end(trial)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_config = load_config("config/train_config_xception.yaml")
    data_config = load_config("config/data_paths.yaml")
    
    epochs = train_config.get("epochs", 20)
    batch_size = train_config.get("batch_size", 64)
    input_shape = tuple(train_config.get("input_shape", [224, 224, 3]))
    train_dir = data_config.get("train_dir", "data/processed/Training")
    validation_split = train_config.get("validation_split", 0.2)
    
    experiments_dir = os.path.abspath(os.path.join(script_dir, "experiments/individual_models/xception3"))
    ensure_dir(experiments_dir)
    
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
        train_dir, target_size=input_shape[:2], batch_size=batch_size, class_mode='binary', subset='training')
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir, target_size=input_shape[:2], batch_size=batch_size, class_mode='binary', subset='validation')
    
    tuner = CustomTuner(
        hypermodel=lambda hp: build_model(hp, input_shape),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=2,
        directory='kt_tuner_dir',
        project_name='xception_tuning3'
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

    tuner.search(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[TrialModelCheckpoint(trial_id=0, base_dir=os.path.join(experiments_dir, "checkpoints")), early_stop, reduce_lr]
    )

    print("Proceso de tuning finalizado. Modelos guardados por trial en 'experiments/individual_models/xception3/checkpoints/'")
