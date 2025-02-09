# 🔥 Detección de Incendios Forestales con Deep Learning 🔥

## 📌 Descripción
Este proyecto forma parte del curso **Deep Learning (1INF52)** de la **Pontificia Universidad Católica del Perú (PUCP)**. Se enfoca en el desarrollo de un modelo basado en aprendizaje profundo para la detección temprana de incendios forestales mediante imágenes capturadas por drones. Se implementa un ensamble de modelos CNN avanzados (Xception, DenseNet y ResNet) y se optimiza utilizando **Knowledge Distillation** y **Pruning** para permitir su implementación en dispositivos con recursos limitados.

## 🎯 **Objetivo del Proyecto**
- Desarrollar un **modelo eficiente y preciso** para la detección de incendios en imágenes.
- Implementar un **ensamble de modelos CNN** para mejorar la generalización.
- Aplicar **Knowledge Distillation** para reducir el tamaño del modelo sin comprometer su precisión.
- Optimizar el modelo final mediante **Pruning** para su implementación en drones.

---

## 📂 **Estructura del Proyecto**
```plaintext
mi_proyecto_fire_detection/
├── data/                  # Almacena datos en diferentes estados
│   ├── train/             # Datos de entrenamiento.
│   └── test/              # Datos de testing/validación.
│
├── experiments/           # Resultados de entrenamiento y evaluación de modelos
│   ├── individual_models/ # Modelos entrenados individualmente (Xception, DenseNet, ResNet)
│   ├── ensemble/          # Experimentos con la fusión de modelos
│   ├── distillation/      # Experimentos de Knowledge Distillation (MobileNetV3)
│   └── pruning/           # Experimentos de optimización de modelo (Pruning)
│
├── notebooks/             # Notebooks para pruebas y análisis exploratorio
│   ├── exploratory.ipynb  # Análisis exploratorio del dataset
│   ├── training.ipynb     # Entrenamiento de modelos
│   └── evaluation.ipynb   # Evaluación de modelos y visualización de métricas
│
├── src/                   # Código fuente organizado por módulos
│   ├── data/              # Scripts para manejo de datos
│   ├── models/            # Definición de modelos (Xception, DenseNet, ResNet, MobileNetV3)
│   ├── training/          # Scripts de entrenamiento para cada modelo
│   ├── evaluation/        # Scripts para evaluación y métricas
│   └── utils/             # Funciones auxiliares y configuraciones
│
├── configs/               # Configuraciones globales del proyecto
│
├── scripts/               # Scripts de ejecución
│   ├── run_training.sh    # Entrenamiento de modelos
│   ├── run_evaluation.sh  # Evaluación de modelos
│
├── requirements.txt       # Dependencias del proyecto
├── README.md              # Este archivo con la documentación del proyecto
└── docs/                  # Documentación y reportes del proyecto
