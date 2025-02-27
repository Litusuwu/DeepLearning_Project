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

# enable GPU optimization
tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy('mixed_float16') # mixed data precision (16-32)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def build_model(hp, input_shape):
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.05, default=0.3)
    l2_factor = hp.Choice('l2_factor', values=[1e-4, 5e-4, 1e-3], default=1e-4)
    n_layers_to_unfreeze = hp.Int('n_layers_to_unfreeze', min_value=10, max_value=50, step=5, default=10)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-4)
    
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    # unfreeze the last n layers
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
    # load configuration files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_config = load_config("config/train_config_xception.yaml")
    data_config = load_config("config/data_paths.yaml")
    
    epochs = train_config.get("epochs", 20)
    batch_size = train_config.get("batch_size", 64)
    input_shape = tuple(train_config.get("input_shape", [224, 224, 3]))
    
    experiments_dir = os.path.abspath(os.path.join(script_dir, "experiments/individual_models/xception_final"))
    logs_dir = os.path.join(experiments_dir, "logs")
    ensure_dir(logs_dir)

    # train and validation data generators
    train_dir = data_config.get("train_dir", "data/raw/Training")
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

    # tuner definition
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(hp, input_shape),
        objective='val_accuracy',
        max_trials=10,  
        executions_per_trial=2,
        directory='keras_tuner',
        project_name='xception_final_tuning'
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="xception_chk.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode='max',
        save_weights_only=False
    )

    early_stopping_config = train_config.get("early_stopping", {})
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor=early_stopping_config.get("monitor", "val_loss"),
        patience=early_stopping_config.get("patience", 5),
        restore_best_weights=True
    )

    rlrop_config = train_config.get("reduce_lr_on_plateau", {})
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=rlrop_config.get("monitor", "val_loss"),
        factor=rlrop_config.get("factor", 0.1),
        patience=rlrop_config.get("patience", 3),
        min_lr=rlrop_config.get("min_lr", 1e-5),
        verbose=1
    )

    callback_list = [checkpoint_callback, early_stopping_cb, reduce_lr_cb]

    if validation_generator is not None:
        tuner.search(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callback_list
        )
    else:
        tuner.search(
            train_generator,
            epochs=epochs,
            callbacks=callback_list
        )

    # print the best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best found hyperparameters:")
    print(f"  - Dropout Rate: {best_hp.get('dropout_rate')}")
    print(f"  - L2 Factor: {best_hp.get('l2_factor')}")
    print(f"  - Number of layers to unfreeze: {best_hp.get('n_layers_to_unfreeze')}")
    print(f"  - Learning Rate: {best_hp.get('learning_rate')}")

    # save the best model
    best_model = tuner.get_best_models(num_models=1)[0]
    model_save_path = os.path.join("keras_tuner/xception_final_tuning", "xcep_final.keras")
    best_model.save(model_save_path, save_format="keras")
    print(f"Model saved: {model_save_path}")