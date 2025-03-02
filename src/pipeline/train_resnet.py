import os
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def build_model(input_shape, dropout_rate, l2_factor, n_layers_to_unfreeze, learning_rate):
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


if __name__ == '__main__':
    train_config = load_config("config/train_config_resnet.yaml")
    data_config = load_config("config/data_paths.yaml")

    epochs = train_config.get("epochs", 20)
    batch_size = train_config.get("batch_size", 128)
    input_shape = tuple(train_config.get("input_shape", [224, 224, 3]))

    best_hp = {
        "dropout_rate": 0.3,
        "l2_factor": 1e-4,
        "n_layers_to_unfreeze": 10,
        "learning_rate": 1e-4
    }

    train_dir = data_config.get("train_dir", "data/processed/Training")
    validation_split = train_config.get("validation_split", 0.0)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
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

    model = build_model(input_shape,
                        best_hp["dropout_rate"],
                        best_hp["l2_factor"],
                        best_hp["n_layers_to_unfreeze"],
                        best_hp["learning_rate"])

    model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    save_dir = "experiments/individual_models/resnet"
    ensure_dir(save_dir)
    model_save_path = os.path.join(save_dir, "resnet_final.keras")
    model.save(model_save_path)
    print(f"ResNet model saved at {model_save_path}")
