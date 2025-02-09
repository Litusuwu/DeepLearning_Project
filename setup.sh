#!/bin/bash
# ----------------------------------------------------------------------
# Este script crea una estructura de directorios para un proyecto
# de Deep Learning, siguiendo una organización básica y "pro".
#
# Uso:
#   ./create_structure.sh <nombre_del_directorio_del_proyecto>
#
# Ejemplo:
#   ./create_structure.sh mi_proyecto_fire_detection
# ----------------------------------------------------------------------

set -e  # Salir inmediatamente si ocurre un error

usage() {
    echo "Uso: $0 <nombre_del_directorio_del_proyecto>"
    exit 1
}

# Verifica que se pase un argumento
if [ -z "$1" ]; then
    echo "Error: No se proporcionó el nombre del directorio del proyecto."
    usage
fi

PROJECT_ROOT="$1"

# Verifica que el nombre no sea solo espacios
if [[ "$PROJECT_ROOT" =~ ^[[:space:]]*$ ]]; then
    echo "Error: El nombre del directorio no puede ser vacío o solo espacios."
    usage
fi

# Validar que el nombre no sea '.' o '..'
if [[ "$PROJECT_ROOT" == "." || "$PROJECT_ROOT" == ".." ]]; then
    echo "Error: '$PROJECT_ROOT' no es un nombre de directorio válido."
    usage
fi

# Comprueba si el directorio ya existe
if [ -d "$PROJECT_ROOT" ]; then
    echo "Error: El directorio '$PROJECT_ROOT' ya existe. Elige otro nombre o elimínalo."
    exit 1
fi

echo "Creando estructura de directorios en '$PROJECT_ROOT'..."

# 1. Crea las carpetas principales
mkdir -p "$PROJECT_ROOT/data/Training"
mkdir -p "$PROJECT_ROOT/data/Test"

mkdir -p "$PROJECT_ROOT/experiments"
mkdir -p "$PROJECT_ROOT/notebooks"
mkdir -p "$PROJECT_ROOT/scripts"

mkdir -p "$PROJECT_ROOT/src/data"
mkdir -p "$PROJECT_ROOT/src/models"
mkdir -p "$PROJECT_ROOT/src/training"
mkdir -p "$PROJECT_ROOT/src/evaluation"
mkdir -p "$PROJECT_ROOT/src/utils"

mkdir -p "$PROJECT_ROOT/configs"

# 2. Archivos y place-holders comunes
touch "$PROJECT_ROOT/requirements.txt"
touch "$PROJECT_ROOT/environment.yml"
touch "$PROJECT_ROOT/README.md"
touch "$PROJECT_ROOT/.gitignore"

# 3. Scripts placeholders
touch "$PROJECT_ROOT/scripts/run_training.sh"
touch "$PROJECT_ROOT/scripts/run_evaluation.sh"

# 4. Notebooks placeholders
touch "$PROJECT_ROOT/notebooks/exploratory.ipynb"
touch "$PROJECT_ROOT/notebooks/training.ipynb"

# 5. Código fuente inicial
touch "$PROJECT_ROOT/src/__init__.py"

# Data
touch "$PROJECT_ROOT/src/data/__init__.py"
touch "$PROJECT_ROOT/src/data/dataset.py"

# Models
touch "$PROJECT_ROOT/src/models/__init__.py"
touch "$PROJECT_ROOT/src/models/densenet_model.py"

# Training
touch "$PROJECT_ROOT/src/training/__init__.py"
touch "$PROJECT_ROOT/src/training/train_densenet.py"

# Evaluation
touch "$PROJECT_ROOT/src/evaluation/__init__.py"
touch "$PROJECT_ROOT/src/evaluation/evaluate.py"

# Utils
touch "$PROJECT_ROOT/src/utils/__init__.py"
touch "$PROJECT_ROOT/src/utils/logger.py"
touch "$PROJECT_ROOT/src/utils/helpers.py"

# 6. Mensaje final
echo "Estructura de directorios creada exitosamente en '$PROJECT_ROOT'."
