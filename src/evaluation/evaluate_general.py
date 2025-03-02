import os
import yaml
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from metrics import compute_metrics, plot_roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_from_weights(config_path, weights_path):
    with open(config_path, "r") as f:
        model_config = json.load(f)

    if "config" in model_config:
        model_config = model_config["config"]

    model = tf.keras.models.model_from_json(json.dumps(model_config))
    model.load_weights(weights_path)  # Load weights
    return model

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    eval_config = load_config("config/eval_config.yaml")
    test_data_cfg = eval_config["test_data"]
    test_dir = test_data_cfg["directory"]
    img_height = test_data_cfg["img_height"]
    img_width = test_data_cfg["img_width"]
    batch_size = test_data_cfg["batch_size"]
    class_mode = test_data_cfg["class_mode"]
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False
    )

    models_info = eval_config["models_to_evaluate"]

    threshold = eval_config["metrics"].get("threshold", 0.5)
    idx_to_class = {v: k for k, v in test_generator.class_indices.items()}
    ordered_classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    log_dir = "logs/evaluation"
    summary_writer = tf.summary.create_file_writer(log_dir)

    conf_matrix_dir = "confusion_matrices"
    os.makedirs(conf_matrix_dir, exist_ok=True)
    for model_info in models_info:
        model_name = model_info["name"]
        model_path = model_info["path"]

        if model_path.endswith(".h5"):
            config_path = os.path.join(os.path.dirname(model_path), "build_config.json")
            if not os.path.exists(config_path):
                print(f"Skipping.")
                continue

            model = load_model_from_weights(config_path, model_path)
            print(f"\n=== Evaluating {model_name} ===")
            print(f"Model loaded from weights: {model_path}")

        elif model_path.endswith(".keras"):
            model = tf.keras.models.load_model(model_path)
            print(f"\n=== Evaluating {model_name} ===")
            print(f"Model loaded from: {model_path}")
        else:
            print(f"Skipping.")
            continue

        steps = int(np.ceil(test_generator.samples / batch_size))
        predictions = model.predict(test_generator, steps=steps, verbose=0)
        true_classes = test_generator.classes

        if class_mode == "binary":
            predictions = predictions.flatten()
        else:
            if predictions.shape[1] == 2:
                predictions = predictions[:, 1]
            else:
                predicted_classes_cat = predictions.argmax(axis=1)
                pass

        metrics_dict, report, roc_data, predicted_classes = compute_metrics(
            true_labels=true_classes,
            predicted_probs=predictions,
            threshold=threshold,
            class_labels=ordered_classes
        )

        with summary_writer.as_default():
            tf.summary.scalar(f"{model_name}/Accuracy", metrics_dict['Accuracy'], step=0)
            tf.summary.scalar(f"{model_name}/F1_Score", metrics_dict['F1 Score'], step=0)
            tf.summary.scalar(f"{model_name}/R2_Score", metrics_dict['R² Score'], step=0)
            tf.summary.scalar(f"{model_name}/AUC_ROC", metrics_dict['AUC-ROC'], step=0)

        print(f"Accuracy: {metrics_dict['Accuracy']:.4f}")
        print(f"F1 Score: {metrics_dict['F1 Score']:.4f}")
        print(f"R² Score: {metrics_dict['R² Score']:.4f}")
        print(f"AUC-ROC: {metrics_dict['AUC-ROC']:.4f}")
        print("Classification Report:")
        print(report)

        fpr, tpr, auc_value = roc_data
        plot_roc_curve(fpr, tpr, auc_value, model_name=model_name)

        conf_matrix = confusion_matrix(true_classes, predicted_classes)

        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=ordered_classes, yticklabels=ordered_classes)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - {model_name}")

        conf_matrix_path = os.path.join(conf_matrix_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(conf_matrix_path)
        plt.close()
        print(f"Confusion matrix saved to: {conf_matrix_path}")

    print("\nEvaluations completed.")
