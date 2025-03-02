import tensorflow as tf

def rename_layers(model, prefix):
    for i, layer in enumerate(model.layers):
        layer.name = f"{prefix}_{layer.name}"

model_densenet = tf.keras.models.load_model("densenet_final.keras", compile=False)
model_resnet   = tf.keras.models.load_model("resnet_final.keras", compile=False)
model_xception = tf.keras.models.load_model("xception_final.keras", compile=False)

rename_layers(model_densenet, "densenet")
rename_layers(model_resnet, "resnet")
rename_layers(model_xception, "xception")

input_layer = tf.keras.layers.Input(shape=(224, 224, 3))

p_densenet = model_densenet(input_layer)
p_resnet   = model_resnet(input_layer)
p_xception = model_xception(input_layer)

avg_prob = tf.keras.layers.Average()([p_densenet, p_resnet, p_xception])

ensemble_model = tf.keras.Model(inputs=input_layer, outputs=avg_prob)

for layer in ensemble_model.layers:
    layer.trainable = False

ensemble_model.save("ensemble_model.keras")
print("Ensemble model saved as ensemble_model.keras")
