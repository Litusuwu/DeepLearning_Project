#!/usr/bin/env python3
import tensorflow as tf
import os


def main():
    print("Cargando modelos pre-entrenados...")

    model_densenet = tf.keras.models.load_model("densenet_final.keras", compile=False)
    model_resnet = tf.keras.models.load_model("resnet_final.keras", compile=False)
    model_xception = tf.keras.models.load_model("xception_final.keras", compile=False)

    input_shape = (224, 224, 3)
    ensemble_input = tf.keras.layers.Input(shape=input_shape, name="ensemble_input")

    def create_submodel(base_model, prefix, input_tensor):
        x = input_tensor

        for i, layer in enumerate(base_model.layers[1:]):
            layer_config = layer.get_config()
            if 'name' in layer_config:
                original_name = layer_config['name']
                layer_config['name'] = f"{prefix}_{original_name}_{i}"

            layer_class = layer.__class__
            new_layer = layer_class.from_config(layer_config)

            if hasattr(layer, 'get_weights') and layer.get_weights():
                new_layer.set_weights(layer.get_weights())

            x = new_layer(x)

        return x

    densenet_path = tf.keras.layers.Lambda(
        lambda x: x,
        name="densenet_preprocessor"
    )(ensemble_input)

    resnet_path = tf.keras.layers.Lambda(
        lambda x: x,
        name="resnet_preprocessor"
    )(ensemble_input)

    xception_path = tf.keras.layers.Lambda(
        lambda x: x,
        name="xception_preprocessor"
    )(ensemble_input)

    densenet_output = tf.keras.models.Model(
        inputs=model_densenet.input,
        outputs=model_densenet.output,
        name="densenet_feature_extractor"
    )(densenet_path)

    resnet_output = tf.keras.models.Model(
        inputs=model_resnet.input,
        outputs=model_resnet.output,
        name="resnet_feature_extractor"
    )(resnet_path)

    xception_output = tf.keras.models.Model(
        inputs=model_xception.input,
        outputs=model_xception.output,
        name="xception_feature_extractor"
    )(xception_path)

    if not os.path.exists("temp_models"):
        os.makedirs("temp_models")

    temp_densenet_path = "temp_models/densenet_outputs.keras"
    temp_resnet_path = "temp_models/resnet_outputs.keras"
    temp_xception_path = "temp_models/xception_outputs.keras"

    tf.keras.models.Model(inputs=model_densenet.input, outputs=model_densenet.output,
                          name="temp_densenet").save(temp_densenet_path)
    tf.keras.models.Model(inputs=model_resnet.input, outputs=model_resnet.output,
                          name="temp_resnet").save(temp_resnet_path)
    tf.keras.models.Model(inputs=model_xception.input, outputs=model_xception.output,
                          name="temp_xception").save(temp_xception_path)

    densenet_model = tf.keras.models.load_model(temp_densenet_path, compile=False)
    resnet_model = tf.keras.models.load_model(temp_resnet_path, compile=False)
    xception_model = tf.keras.models.load_model(temp_xception_path, compile=False)

    densenet_model._name = "densenet_output_model"
    resnet_model._name = "resnet_output_model"
    xception_model._name = "xception_output_model"

    densenet_preds = densenet_model(ensemble_input)
    resnet_preds = resnet_model(ensemble_input)
    xception_preds = xception_model(ensemble_input)

    ensemble_output = tf.keras.layers.Average(name="ensemble_average")(
        [densenet_preds, resnet_preds, xception_preds]
    )

    ensemble_model = tf.keras.models.Model(
        inputs=ensemble_input,
        outputs=ensemble_output,
        name="ensemble_model"
    )

    from tensorflow.keras.utils import plot_model
    plot_model(ensemble_model, show_shapes=True, to_file="ensemble_model.png")

    ensemble_model.save("ensemble_model2.keras")
    print("Modelo ensemble guardado exitosamente en ensemble_model2.keras")

    for temp_file in [temp_densenet_path, temp_resnet_path, temp_xception_path]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    if os.path.exists("temp_models"):
        os.rmdir("temp_models")


if __name__ == "__main__":
    main()