import os
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def build_resnet_model(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=predictions)

if __name__ == '__main__':
    train_config = load_config("configs/train_config_resnet.yaml")
    data_config = load_config("configs/data_paths.yaml")

    epochs = train_config.get("epochs", 10)
    batch_size = train_config.get("batch_size", 32)
    input_shape = tuple(train_config.get("input_shape", [224, 224, 3]))
    optimizer = train_config.get("optimizer", "adam")
    loss = train_config.get("loss", "binary_crossentropy")
    metrics_list = train_config.get("metrics", ["accuracy"])

    model_name = "resnet"
    logs_dir = train_config["output_dirs"]["logs"]
    checkpoints_dir = train_config["output_dirs"]["checkpoints"]
    results_dir = train_config["output_dirs"]["results"]
    ensure_dir(logs_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(results_dir)

    train_dir = data_config.get("train_dir", "data/processed/Training")
    val_dir = data_config.get("val_dir", None)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=train_config.get("augmentation", {}).get("shear_range", 0.0),
        zoom_range=train_config.get("augmentation", {}).get("zoom_range", 0.0),
        horizontal_flip=train_config.get("augmentation", {}).get("horizontal_flip", False),
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

    model = build_resnet_model(input_shape)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics_list)

    checkpoint_path = os.path.join(checkpoints_dir, "resnet_best.h5")
    csv_logger_path = os.path.join(logs_dir, "resnet_training.csv")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor=train_config.get("early_stopping", {}).get("monitor", "val_loss"),
                                 save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor=train_config.get("early_stopping", {}).get("monitor", "val_loss"),
                                   patience=train_config.get("early_stopping", {}).get("patience", 3),
                                   verbose=1)
    csv_logger = CSVLogger(csv_logger_path)
    callbacks_list = [checkpoint, early_stopping, csv_logger]

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

    final_model_path = os.path.join(results_dir, "resnet_final.h5")
    model.save(final_model_path)
    print("Modelo ResNet final guardado en:", final_model_path)
