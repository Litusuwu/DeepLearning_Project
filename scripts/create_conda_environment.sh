#!/bin/bash
# -----------------------------------------
# create_environment.sh
# Este script crea (o actualiza) un entorno
# conda usando environment.yml
# -----------------------------------------
# Crea/actualiza el entorno usando environment.yml
# ./scripts/create_environment.sh

# # Activa el entorno
# conda activate fire_detection_env

# # Instala (o reinstala) paquetes de requirements.txt
# ./scripts/install_requirements.sh

# Nombre de tu entorno (puedes leerlo de environment.yml o definirlo aquí)
ENV_NAME="fire_detection_env"

echo "Creando (o actualizando) el entorno conda '$ENV_NAME' usando environment.yml..."

# Verifica que conda esté instalado
if ! command -v conda &> /dev/null
then
    echo "Error: conda no está instalado o no está en el PATH."
    exit 1
fi

# Crea o actualiza el entorno a partir de environment.yml
conda env create -f environment.yml --force

# Activa el entorno para confirmar la instalación
echo "Activando el entorno '$ENV_NAME'..."
conda activate "$ENV_NAME"

echo "¡Entorno conda '$ENV_NAME' creado exitosamente!"
echo "Ahora puedes usar: conda activate $ENV_NAME"
