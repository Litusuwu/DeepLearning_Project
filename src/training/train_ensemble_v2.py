#!/usr/bin/env python3
import tensorflow as tf
import os

def main():
    print("Cargando modelos pre-entrenados...")
    
    # Cargar modelos individuales
    model_densenet = tf.keras.models.load_model("densenet_final.keras", compile=False)
    model_resnet = tf.keras.models.load_model("resnet_final.keras", compile=False)
    model_xception = tf.keras.models.load_model("xception_final.keras", compile=False)
    
    # Definir entrada compartida para el ensemble
    input_shape = (224, 224, 3)
    ensemble_input = tf.keras.layers.Input(shape=input_shape, name="ensemble_input")
    
    # ===== ENFOQUE 1: EXTRAER LOS MODELOS COMO "CAJAS NEGRAS" =====
    # En lugar de usar los modelos directamente, extraemos su funcionalidad
    # creando nuevos modelos con las mismas capas pero nombres diferentes
    
    # Función para crear submodelos con nombres únicos
    def create_submodel(base_model, prefix, input_tensor):
        # Crear una versión con nombres únicos
        x = input_tensor
        
        # Copiar la estructura del modelo base, capa por capa con nombres únicos
        for i, layer in enumerate(base_model.layers[1:]):  # Saltamos la capa de entrada
            # Crear una capa con la misma configuración pero nombre único
            layer_config = layer.get_config()
            if 'name' in layer_config:
                original_name = layer_config['name']
                layer_config['name'] = f"{prefix}_{original_name}_{i}"
            
            # Crear una nueva instancia de la capa
            layer_class = layer.__class__
            new_layer = layer_class.from_config(layer_config)
            
            # Copiar los pesos si existen
            if hasattr(layer, 'get_weights') and layer.get_weights():
                new_layer.set_weights(layer.get_weights())
            
            # Aplicar la capa
            x = new_layer(x)
        
        return x
    
    # ===== ENFOQUE 2: USAR ARQUITECTURA DE SALTO (BYPASS) =====
    # Otra estrategia es evitar por completo el problema de los nombres funcionales
    # utilizando submodelos independientes conectados con entradas/salidas separadas
    
    # Creamos rutas independientes para cada modelo
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
    
    # Extraemos cada predicción
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
    
    # ===== ENFOQUE 3: MODELO SECUENCIAL =====
    # Si los enfoques anteriores no funcionan, podemos intentar usar modelos secuenciales
    # que agrupan las capas de forma más simple
    
    # Guardar modelos individuales en formatos separados para cargarlos luego
    if not os.path.exists("temp_models"):
        os.makedirs("temp_models")
    
    temp_densenet_path = "temp_models/densenet_outputs.keras"
    temp_resnet_path = "temp_models/resnet_outputs.keras"
    temp_xception_path = "temp_models/xception_outputs.keras"
    
    # Guardamos modelos intermedios de salida
    tf.keras.models.Model(inputs=model_densenet.input, outputs=model_densenet.output, 
                          name="temp_densenet").save(temp_densenet_path)
    tf.keras.models.Model(inputs=model_resnet.input, outputs=model_resnet.output, 
                          name="temp_resnet").save(temp_resnet_path)
    tf.keras.models.Model(inputs=model_xception.input, outputs=model_xception.output, 
                          name="temp_xception").save(temp_xception_path)
    
    # Cargamos esos modelos como objetos separados
    densenet_model = tf.keras.models.load_model(temp_densenet_path, compile=False)
    resnet_model = tf.keras.models.load_model(temp_resnet_path, compile=False)
    xception_model = tf.keras.models.load_model(temp_xception_path, compile=False)
    
    # Renombramos cada modelo para garantizar que no haya conflictos
    densenet_model._name = "densenet_output_model"
    resnet_model._name = "resnet_output_model"
    xception_model._name = "xception_output_model"
    
    # Aplicamos cada modelo al input
    densenet_preds = densenet_model(ensemble_input)
    resnet_preds = resnet_model(ensemble_input)
    xception_preds = xception_model(ensemble_input)
    
    # Combinamos las predicciones con promedio
    ensemble_output = tf.keras.layers.Average(name="ensemble_average")(
        [densenet_preds, resnet_preds, xception_preds]
    )
    
    # Creamos el modelo final
    ensemble_model = tf.keras.models.Model(
        inputs=ensemble_input,
        outputs=ensemble_output,
        name="ensemble_model"
    )

    from tensorflow.keras.utils import plot_model
    plot_model(ensemble_model, show_shapes=True, to_file="ensemble_model.png")
    
    # Guardamos el modelo
    ensemble_model.save("ensemble_model2.keras")
    print("✅ Modelo ensemble guardado exitosamente en ensemble_model2.keras")
    
    # Limpieza de archivos temporales
    for temp_file in [temp_densenet_path, temp_resnet_path, temp_xception_path]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    if os.path.exists("temp_models"):
        os.rmdir("temp_models")

if __name__ == "__main__":
    main()