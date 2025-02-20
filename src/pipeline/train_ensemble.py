import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model, clone_model
from tensorflow.keras.layers import Average, Input


def rename_model_layers(model, prefix):
    """Recursively rename all layers in the model and its submodels."""
    # Keep track of used names to ensure uniqueness
    used_names = set()

    def get_unique_name(name, prefix):
        """Generate a unique name by adding a suffix if necessary."""
        new_name = f"{prefix}_{name}"
        counter = 1
        while new_name in used_names:
            new_name = f"{prefix}_{name}_{counter}"
            counter += 1
        used_names.add(new_name)
        return new_name

    def rename_layers(layer):
        """Recursively rename layers."""
        # Skip if this is an input layer
        if isinstance(layer, tf.keras.layers.InputLayer):
            return

        # Give the layer a unique name
        layer._name = get_unique_name(layer.name, prefix)

        # If this layer contains other layers, rename them too
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                rename_layers(sublayer)

    # Start the recursive renaming
    for layer in model.layers:
        rename_layers(layer)

    return model


def clone_with_unique_names(model, prefix):
    """Clone a model and ensure all layers have unique names."""
    cloned = clone_model(model)
    cloned._name = f"{prefix}_model"

    # Recursively rename all layers
    cloned = rename_model_layers(cloned, prefix)

    # Copy weights
    cloned.set_weights(model.get_weights())
    return cloned


densenet_path = "experiments/individual_models/densenet/densenet_final.keras"
resnet_path = "experiments/individual_models/resnet/resnet_final.keras"
xception_path = "experiments/individual_models/xception/xception_final.keras"

model_densenet = load_model(densenet_path)
model_resnet = load_model(resnet_path)
model_xception = load_model(xception_path)

# Clone each model with a unique name
model_densenet = clone_with_unique_names(model_densenet, "densenet")
model_resnet   = clone_with_unique_names(model_resnet,   "resnet")
model_xception = clone_with_unique_names(model_xception, "xception")

# freeze the models
model_densenet.trainable = False
model_resnet.trainable = False
model_xception.trainable = False

# Assume all models share the same input shape
input_shape = model_densenet.input_shape[1:]
inp = Input(shape=input_shape, name="ensemble_input")

# Use separate name scopes to avoid duplicate operation names
with tf.name_scope("densenet_scope"):
    pred1 = model_densenet(inp)
with tf.name_scope("resnet_scope"):
    pred2 = model_resnet(inp)
with tf.name_scope("xception_scope"):
    pred3 = model_xception(inp)

# average the predictions
ensemble_output = Average(name="ensemble_average")([pred1, pred2, pred3])
ensemble_model = Model(inputs=inp, outputs=ensemble_output, name="ensemble_model")
ensemble_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# save the ensemble model
save_path = "experiments/individual_models/ensemble_model.keras"
ensemble_model.save(save_path)
print(f"✅ Ensemble model saved at {save_path}")
