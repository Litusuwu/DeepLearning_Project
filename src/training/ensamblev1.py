#!/usr/bin/env python3
import tensorflow as tf

def main():
    print("Cargando modelos pre-entrenados...")
    model_densenet = tf.keras.models.load_model("densenet_final.keras", compile=False)
    model_resnet = tf.keras.models.load_model("resnet_final.keras", compile=False)
    model_xception = tf.keras.models.load_model("xception_final.keras", compile=False)
    
    
    input_shape = (224, 224, 3)
    ensemble_input = tf.keras.layers.Input(shape=input_shape, name="ensemble_input")
    
    densenet_input = tf.keras.layers.Input(shape=input_shape, name="densenet_input")
    densenet_model = tf.keras.models.Model(
        inputs=model_densenet.input, 
        outputs=model_densenet.output,
        name="densenet_feature_extractor"
    )
    densenet_output = densenet_model(ensemble_input)
    
    resnet_input = tf.keras.layers.Input(shape=input_shape, name="resnet_input")
    resnet_model = tf.keras.models.Model(
        inputs=model_resnet.input, 
        outputs=model_resnet.output,
        name="resnet_feature_extractor"
    )
    resnet_output = resnet_model(ensemble_input)
    
    xception_input = tf.keras.layers.Input(shape=input_shape, name="xception_input")
    xception_model = tf.keras.models.Model(
        inputs=model_xception.input, 
        outputs=model_xception.output,
        name="xception_feature_extractor"
    )
    xception_output = xception_model(ensemble_input)
    
    ensemble_output = tf.keras.layers.Average(name="ensemble_average")(
        [densenet_output, resnet_output, xception_output]
    )
    
    ensemble_model = tf.keras.models.Model(
        inputs=ensemble_input,
        outputs=ensemble_output,
        name="ensemble_final"
    )
    
    print("Guardando modelo ensemble")
    ensemble_model.save("ensemble_modelv1.keras")
    print("Modelo ensemble guardado en ensemble_modelv1.keras")

if __name__ == "__main__":
    main()