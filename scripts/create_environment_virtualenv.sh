#!/bin/bash
# -----------------------------------------
# create_environment.sh
# Crea un virtualenv e instala paquetes
# desde environment.yml o requirements.txt
# -----------------------------------------
# Crea el entorno virtual y lo activa
# ./scripts/create_environment.sh

# # (La secuencia ya instala los paquetes)
# # Si quieres reinstalar en otro momento:
# source fire_detection_env/bin/activate
# ./scripts/install_requirements.sh

ENV_NAME="fire_detection_env"

# Crea el virtualenv
echo "Creando virtualenv '$ENV_NAME'..."
python3 -m venv "$ENV_NAME"

# Activa el virtualenv
echo "Activando virtualenv '$ENV_NAME'..."
source "$ENV_NAME/bin/activate"

# Instala dependencias (si quisieras usar environment.yml, tendrías que parsearlo
# o convertirlo manualmente. Con pip, es más directo usar requirements.txt)
pip install --upgrade pip
pip install -r requirements.txt

echo "¡Virtualenv '$ENV_NAME' creado y paquetes instalados!"
echo "Para activar en el futuro: source $ENV_NAME/bin/activate"
