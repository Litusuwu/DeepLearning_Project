import os
import json
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

tf.config.optimizer.set_jit(True)  # Habilitar XLA (Aceleración)
tf.keras.mixed_precision.set_global_policy('mixed_float16')  # Usar mixed precision

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
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
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


# Custom callback to save each trial's best model in its own directory
class TrialModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, trial_id, base_dir):
        super(TrialModelCheckpoint, self).__init__()
        self.trial_id = trial_id
        self.base_dir = base_dir
        self.best_val_loss = float('inf')
        self.trial_dir = os.path.join(self.base_dir, f"trial_{self.trial_id:02d}")
        ensure_dir(self.trial_dir)
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss")
        if val_loss and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            model_save_path = os.path.join(self.trial_dir, "best_model.keras")
            self.model.save(model_save_path)
            print(f"Trial {self.trial_id:02d}: Saved new best model with val_loss {val_loss:.4f} at epoch {epoch}")




if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_config = load_config("config/train_config_resnet.yaml")
    data_config = load_config("config/data_paths.yaml")
    
    epochs = train_config.get("epochs", 20)
    batch_size = train_config.get("batch_size", 128)
    input_shape = tuple(train_config.get("input_shape", [224, 224, 3]))
    
    experiments_dir = os.path.abspath(os.path.join(script_dir, "experiments/individual_models/resnet"))
    logs_dir = os.path.join(experiments_dir, "logs")
    ensure_dir(logs_dir)

    train_dir = data_config.get("train_dir", "data/processed/Training")
    validation_split = train_config.get("validation_split", 0.0)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=train_config.get("augmentation", {}).get("shear_range", 0.0),
        zoom_range=train_config.get("augmentation", {}).get("zoom_range", 0.0),
        horizontal_flip=train_config.get("augmentation", {}).get("horizontal_flip", False),
        rotation_range=train_config.get("augmentation", {}).get("rotation_range", 10),
        brightness_range=train_config.get("augmentation", {}).get("brightness_range", [0.8, 1.2]),
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

    # Definir el tuner
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(hp, input_shape),
        objective='val_accuracy',
        max_trials=10,  
        executions_per_trial=2,
        directory='kt_tuner_dir',
        project_name='resnet_tuning'
    )

    # Run tuner.search() once so that all trials are managed internally.
    tuner.search(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[
            # Here we add the custom callback. You can retrieve the trial number from tuner.get_trial() if needed.
            # For illustration, we attach a callback that saves to a common directory.
            TrialModelCheckpoint(trial_id=0, base_dir=os.path.join(experiments_dir, "checkpoints")),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=train_config.get("early_stopping", {}).get("patience", 5))
        ]
    )

    # Save the best overall model and hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)
    best_model_path = os.path.join(experiments_dir, "final_best_model.keras")
    best_model.save(best_model_path)
    print(f"✅ Best overall model saved: {best_model_path}")

    # Optionally, save the best hyperparameters to a JSON file for record keeping
    best_hp_config = {param: best_hp.get(param) for param in best_hp.values.keys()}
    with open(os.path.join(experiments_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_hp_config, f, indent=4)
    print("✅ Best hyperparameters saved.")