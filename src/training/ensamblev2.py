#!/usr/bin/env python3
import tensorflow as tf

def extract_backbone(model, prefix):
    """Extracts the backbone from a model by recreating each layer with new names"""
    layers = []
    for i, layer in enumerate(model.layers):
        config = layer.get_config()
        # Modify the name to ensure uniqueness
        if 'name' in config:
            config['name'] = f"{prefix}_{config['name']}_{i}"
        # Create a new layer with the updated config
        if hasattr(layer, 'get_weights'):
            new_layer = layer.__class__.from_config(config)
            weights = layer.get_weights()
            if weights:
                new_layer.set_weights(weights)
            layers.append(new_layer)
        else:
            layers.append(layer.__class__.from_config(config))
    return layers

def main():
    # 1) Cargar los modelos pre-entrenados (sin compilar)
    print("Cargando modelos pre-entrenados...")
    model_densenet = tf.keras.models.load_model("densenet_final.keras", compile=False)
    model_resnet = tf.keras.models.load_model("resnet_final.keras", compile=False)
    model_xception = tf.keras.models.load_model("xception_final.keras", compile=False)
    
    # 2) Crear un input compartido
    input_shape = (224, 224, 3)
    ensemble_input = tf.keras.layers.Input(shape=input_shape, name="ensemble_input")
    
    # 3) Extraer los features con cada modelo
    # Usamos capas funcionales para evitar problemas con nombres duplicados
    
    # Crear extractores de características completamente nuevos
    densenet_features = model_densenet(ensemble_input, training=False)
    densenet_output = tf.keras.layers.Lambda(lambda x: x, name="densenet_output")(densenet_features)
    
    resnet_features = model_resnet(ensemble_input, training=False)
    resnet_output = tf.keras.layers.Lambda(lambda x: x, name="resnet_output")(resnet_features)
    
    xception_features = model_xception(ensemble_input, training=False)
    xception_output = tf.keras.layers.Lambda(lambda x: x, name="xception_output")(xception_features)
    
    # 4) Combinar utilizando una capa promedio
    ensemble_output = tf.keras.layers.Average(name="ensemble_average")(
        [densenet_output, resnet_output, xception_output]
    )
    
    # 5) Construir el modelo ensemble final
    ensemble_model = tf.keras.models.Model(
        inputs=ensemble_input,
        outputs=ensemble_output,
        name="ensemble_model"
    )
    
    # 6) Guardar el modelo ensemble
    print("Guardando modelo ensemble...")
    ensemble_model.save("ensemble_modelv2.keras")
    print("✅ Modelo ensemble guardado exitosamente en ensemble_modelv2.keras")

if __name__ == "__main__":
    main()