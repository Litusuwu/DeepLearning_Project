#!/bin/bash
# run_pipeline.sh: Execute the training, ensembling, and distillation pipeline

echo "Starting pipeline..."

# 1. train DenseNet with best hyperparameters
echo "Training DenseNet..."
python src/pipeline/train_densenet.py

# 2. train ResNet with best hyperparameters
echo "Training ResNet..."
python src/pipeline/train_resnet.py

# 3. train Xception with best hyperparameters
echo "Training Xception..."
python src/pipeline/train_xception.py

# 4. ensemble the three models
echo "Building ensemble model..."
python src/pipeline/train_ensemble.py

# 5. distill the ensemble into a MobileNetV3
echo "Performing model distillation..."
python src/pipeline/train_distillation.py

echo "Pipeline complete!"
