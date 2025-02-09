#!/bin/bash
# --------------------------------------------
# install_requirements.sh
# Instala dependencias desde requirements.txt
# --------------------------------------------

echo "Instalando dependencias desde requirements.txt..."

# Verifica si pip está disponible en el entorno
if ! command -v pip &> /dev/null
then
    echo "Error: pip no está disponible. ¿Olvidaste activar tu entorno?"
    exit 1
fi

pip install --upgrade pip
pip install -r requirements.txt

echo "¡Dependencias instaladas exitosamente!"
