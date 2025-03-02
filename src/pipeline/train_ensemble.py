import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model, clone_model
from tensorflow.keras.layers import Average, Input


def rename_model_layers(model, prefix):
    used_names = set()

    def get_unique_name(name, prefix):
        new_name = f"{prefix}_{name}"
        counter = 1
        while new_name in used_names:
            new_name = f"{prefix}_{name}_{counter}"
            counter += 1
        used_names.add(new_name)
        return new_name

    def rename_layers(layer):
        if isinstance(layer, tf.keras.layers.InputLayer):
            return

        layer._name = get_unique_name(layer.name, prefix)

        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                rename_layers(sublayer)

    for layer in model.layers:
        rename_layers(layer)

    return model


def clone_with_unique_names(model, prefix):
    cloned = clone_model(model)
    cloned._name = f"{prefix}_model"

    cloned = rename_model_layers(cloned, prefix)

    cloned.set_weights(model.get_weights())
    return cloned


densenet_path = "experiments/individual_models/densenet/densenet_final.keras"
resnet_path = "experiments/individual_models/resnet/resnet_final.keras"
xception_path = "experiments/individual_models/xception/xception_final.keras"

model_densenet = load_model(densenet_path)
model_resnet = load_model(resnet_path)
model_xception = load_model(xception_path)

model_densenet = clone_with_unique_names(model_densenet, "densenet")
model_resnet   = clone_with_unique_names(model_resnet,   "resnet")
model_xception = clone_with_unique_names(model_xception, "xception")

model_densenet.trainable = False
model_resnet.trainable = False
model_xception.trainable = False

input_shape = model_densenet.input_shape[1:]
inp = Input(shape=input_shape, name="ensemble_input")

with tf.name_scope("densenet_scope"):
    pred1 = model_densenet(inp)
with tf.name_scope("resnet_scope"):
    pred2 = model_resnet(inp)
with tf.name_scope("xception_scope"):
    pred3 = model_xception(inp)

ensemble_output = Average(name="ensemble_average")([pred1, pred2, pred3])
ensemble_model = Model(inputs=inp, outputs=ensemble_output, name="ensemble_model")
ensemble_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

save_path = "experiments/individual_models/ensemble_model.keras"
ensemble_model.save(save_path)
print(f"✅ Ensemble model saved at {save_path}")
