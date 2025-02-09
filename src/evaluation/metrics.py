import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    roc_curve,
    auc,
    classification_report
)

def compute_metrics(true_labels, predicted_probs, threshold=0.5, class_labels=None):
    """
    Calcula métricas de evaluación estándar para clasificación binaria.

    Parámetros:
    ----------
    - true_labels : numpy array
        Etiquetas reales (0 o 1).
    - predicted_probs : numpy array
        Probabilidades predichas por el modelo (entre 0 y 1).
    - threshold : float, opcional (default=0.5)
        Umbral para convertir probabilidades en etiquetas binarias.
    - class_labels : list or tuple, opcional
        Nombres de las clases (p. ej. ["NoFire", "Fire"]). 
        Si no se pasa, se usará ["Class 0", "Class 1"].

    Retorna:
    --------
    - metrics_dict : dict
        Diccionario con 'Accuracy', 'F1 Score', 'R² Score' y 'AUC-ROC'.
    - report : str
        Reporte de clasificación (precision, recall, f1-score, support).
    - roc_data : tuple (fpr, tpr, auc_value)
        Datos para graficar la curva ROC (False Positive Rate, True Positive Rate, AUC).
    - predicted_classes : numpy array
        Etiquetas binarias predichas (0 o 1).
    """
    if class_labels is None:
        class_labels = ["Class 0", "Class 1"]

    # Convertir probabilidades a clases binarias
    predicted_classes = (predicted_probs > threshold).astype("int32").flatten()

    # Accuracy y F1 Score
    accuracy = accuracy_score(true_labels, predicted_classes)
    f1 = f1_score(true_labels, predicted_classes)

    # R² Score (usamos las probabilidades sin binarizar)
    r2 = r2_score(true_labels, predicted_probs)

    # Curva ROC y AUC-ROC
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    auc_value = auc(fpr, tpr)

    # Reporte de clasificación
    report = classification_report(true_labels, predicted_classes, target_names=class_labels)

    # Guardamos las métricas en un diccionario
    metrics_dict = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "R² Score": r2,
        "AUC-ROC": auc_value
    }

    return metrics_dict, report, (fpr, tpr, auc_value), predicted_classes

def plot_roc_curve(fpr, tpr, auc_value, model_name="Model"):
    """
    Genera e imprime la curva ROC.

    Parámetros:
    -----------
    - fpr : array
        False Positive Rate en distintos umbrales.
    - tpr : array
        True Positive Rate en distintos umbrales.
    - auc_value : float
        Área bajo la curva ROC.
    - model_name : str, opcional
        Nombre del modelo para el título de la gráfica.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()
