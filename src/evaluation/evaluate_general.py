import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Para métricas y curva ROC
from metrics import compute_metrics, plot_roc_curve

def load_config(config_path: str) -> dict:
    """Carga un archivo YAML y retorna un diccionario con la configuración."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # 1. Cargar la configuración de evaluación
    eval_config = load_config("config/eval_config.yaml")

    # 2. Extraer parámetros de test_data
    test_data_cfg = eval_config["test_data"]
    test_dir = test_data_cfg["directory"]
    img_height = test_data_cfg["img_height"]
    img_width = test_data_cfg["img_width"]
    batch_size = test_data_cfg["batch_size"]
    class_mode = test_data_cfg["class_mode"]

    # 3. setup ImageDataGenerator for testing
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False  # to align predictions w the right tags
    )

    # 4. Cargar la lista de modelos a evaluar
    models_info = eval_config["models_to_evaluate"]

    # 5. Configuración adicional de métricas
    threshold = eval_config["metrics"].get("threshold", 0.5)

    # 6. Determinar etiquetas para el reporte
    #    Si es binario con 2 clases, generador tiene algo como {'fire': 0, 'nofire': 1} 
    #    Vamos a obtenerlos en orden de indices:
    idx_to_class = {v: k for k, v in test_generator.class_indices.items()}
    # Ejemplo: 0 -> 'fire', 1 -> 'nofire'
    ordered_classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    # 7. Evaluar cada modelo
    for model_info in models_info:
        model_name = model_info["name"]
        model_path = model_info["path"]

        if model_path.endswith(".keras") or model_path.endswith(".h5"):
            model = tf.keras.models.load_model(model_path) # load model
            print(f"\n=== Evaluating {model_name} ===")
            print(f"Model in: {model_path}")
        else:
            print(f"⚠️ Unsupported model format: {model_path}. Skipping.")
            continue

        # predictions set
        steps = int(np.ceil(test_generator.samples / batch_size))
        predictions = model.predict(test_generator, steps=steps, verbose=0)

        # real tags
        true_classes = test_generator.classes

        # (Opcional) Ajuste si class_mode es 'categorical' vs 'binary'
        # En este ejemplo, asumo 'binary', si 'categorical', habría que usar argmax
        if class_mode == "binary":
            # Convertimos a forma (N,) en vez de (N,1)
            predictions = predictions.flatten()
        else:
            # Para 'categorical'
            # predictions shape: (N, num_clases)
            # Tomamos las probabilidades de la clase "positiva" si es binario 
            # o hacemos argmax si es multi-clase
            if predictions.shape[1] == 2:
                # asumiendo que la clase 1 es 'positiva'
                predictions = predictions[:, 1]
            else:
                # argmax multi-clase
                predicted_classes_cat = predictions.argmax(axis=1)
                # Pasar predicted_classes_cat a la función compute_metrics con threshold = -1
                # o en realidad, habría que crear otra función de métricas para multi-clase
                # ... se simplifica en binario en este ejemplo
                pass

        # 8. calculate metrics
        metrics_dict, report, roc_data, predicted_classes = compute_metrics(
            true_labels=true_classes,
            predicted_probs=predictions,
            threshold=threshold,
            class_labels=ordered_classes
        )

        # 9. Mostrar métricas
        print(f"Accuracy: {metrics_dict['Accuracy']:.4f}")
        print(f"F1 Score: {metrics_dict['F1 Score']:.4f}")
        print(f"R² Score: {metrics_dict['R² Score']:.4f}")
        print(f"AUC-ROC: {metrics_dict['AUC-ROC']:.4f}")
        print("Classification Report:")
        print(report)

        # 10. Graficar la curva ROC
        fpr, tpr, auc_value = roc_data
        plot_roc_curve(fpr, tpr, auc_value, model_name=model_name)

    print("\nEvaluations completed.")
