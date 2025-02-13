import os
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np
from datetime import datetime

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPU Available:", gpus)
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("No GPU detected, using CPU.")

# utility functions
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def build_densenet_model(input_shape):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # freeze the base model

    for i, layer in enumerate(base_model.layers[-5:]):
        layer.trainable = True
        print(f"Unfreezing Layer {i+1}: {layer.name}")

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)

    predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(x)
    return Model(inputs=base_model.input, outputs=predictions)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_config = load_config("config/train_config_densenet.yaml")
    data_config = load_config("config/data_paths.yaml")

    epochs = train_config.get("epochs", 10)
    batch_size = train_config.get("batch_size", 8)
    input_shape = tuple(train_config.get("input_shape", [224, 224, 3]))
    optimizer = Adam(learning_rate=0.0001)
    loss = train_config.get("loss", "binary_crossentropy")
    metrics_list = train_config.get("metrics", ["accuracy"])

    experiments_dir = os.path.abspath(os.path.join(script_dir, "experiments/individual_models/densenet"))

    logs_dir = os.path.join(experiments_dir, "logs")
    checkpoints_dir = os.path.join(experiments_dir, "checkpoints")
    results_dir = os.path.join(experiments_dir, "results")

    ensure_dir(logs_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(results_dir)

    # Configurar rutas de datos
    train_dir = data_config.get("train_dir", "data/processed/Training")
    val_dir = data_config.get("val_dir", None)

    # Generadores de datos (usando validation_split si no hay carpeta de validación)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=train_config.get("augmentation", {}).get("shear_range", 0.0),
        zoom_range=train_config.get("augmentation", {}).get("zoom_range", 0.0),
        horizontal_flip=train_config.get("augmentation", {}).get("horizontal_flip", False),
        rotation_range=20,
        validation_split=train_config.get("validation_split", 0.0)
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

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

    # build and compile the model
    model = build_densenet_model(input_shape)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics_list)

    # Callbacks
    checkpoint_path = os.path.join(checkpoints_dir, "densenet_best.h5")
    csv_logger_path = os.path.join(logs_dir, "densenet_training.csv")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor=train_config.get("early_stopping", {}).get("monitor", "val_loss"),
                                 save_best_only=True, verbose=1)

    early_stopping_monitor = train_config.get("early_stopping", {}).get("monitor", "val_loss")
    early_stopping_patience = train_config.get("early_stopping", {}).get("patience", 3)

    early_stopping = EarlyStopping(monitor=early_stopping_monitor, patience=early_stopping_patience, verbose=1)

    csv_logger = CSVLogger(csv_logger_path)
    callbacks_list = [checkpoint, early_stopping, csv_logger]

    # training
    if validation_generator is not None:

        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_generator.classes)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        history = model.fit(
            train_generator,
            steps_per_epoch=max(1, train_generator.samples // batch_size),
            validation_data=validation_generator,
            validation_steps=max(1, validation_generator.samples // batch_size),
            epochs=epochs,
            class_weight=class_weight_dict,
            callbacks=callbacks_list
        )
    else:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            callbacks=callbacks_list
        )

    # save the final model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"model_{timestamp}.keras"
    final_model_path = os.path.join(results_dir, model_name)
    model.save(final_model_path)
    print("DenseNet Model saved on:", final_model_path)
