# Wildfire Detection with Deep Learning

<p align="center">
    <img src="https://img.shields.io/badge/Python-f9e2af?logo=python&logoColor=black" alt="Python" />
    <img src="https://img.shields.io/badge/TensorFlow-f2cdcd?logo=tensorflow&logoColor=black" alt="TensorFlow" />
</p>

This project, part of the **Deep Learning (1INF52)** course at **PUCP**, aims to create a deep learning model
for early forest fire detection using drone images. We combine three CNN models (Xception, DenseNet, ResNet)
to improve generalization and achieve top performance. The model is trained on the
[FLAME dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs),
which contains aerial images of controlled burns in Arizona’s pine forests.

## 🎯 **Objectives**
- [x] Develop an efficient and accurate fire detection model for images (hopefully that reaches SOTA).
- [x] Implement a CNN model ensemble to improve generalization.
- [ ] (Future work) Optimize the final model with pruning for drone deployment.
---

## 📂 **Project Structure**
```plaintext
mi_proyecto_fire_detection/
├── data/                  # stores datasets (test + training data)
│   ├── train/
│   └── test/
│
├── notebooks/             # tsting and exploratory analysis notebooks
│   ├── exploratory.ipynb
│   ├── training.ipynb
│   └── evaluation.ipynb
│
├── src/                   # source code
│   ├── data/              # scripts for data handling
│   ├── pipeline/          # code for pipeline execution (with already set parameters)
│   ├── training/
│   ├── evaluation/
│   └── utils/             # additional handlers
│
├── configs/               # global config files
│
├── scripts/               # execution scripts
│   ├── run_training.sh
│   ├── run_evaluation.sh
│
├── requirements.txt
├── README.md
└── report/                # project final report
```

## Web App and Hugging Face

The trained models can be tested with a [simple web app](https://github.com/superflash41/isaFIRE) built
to interact with it.

They can also be found and used on [Hugging Face](https://huggingface.co/superflash41/fire-chad-detector-v1.0).

## Documentation

The project's final report with the detailed explanation of our research can be found in the [`report/`](report) folder
and the presentation and poster can be found on the [`presentation/`](presentation) folder.
