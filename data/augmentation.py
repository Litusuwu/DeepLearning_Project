from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import math
from PIL import Image

# Configuración de aumentación
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Rutas: asegúrate de que estas rutas sean las correctas en tu estructura de carpetas
no_fire_dir = "data/processed/Training/No_Fire_Resized"
output_dir = "data/processed/Training/No_Fire"
os.makedirs(output_dir, exist_ok=True)

# Número total deseado de imágenes para la clase "No_Fire"
target_no_fire = 50000 
# Cantidad de imágenes originales en la carpeta
current_no_fire = len(os.listdir(no_fire_dir))
# Número de imágenes adicionales que queremos generar
images_to_generate = target_no_fire - current_no_fire

if current_no_fire == 0:
    raise ValueError(f"No se encontraron imágenes en {no_fire_dir}.")

# Calcula cuántas augmentaciones se deben generar por cada imagen original
augmentations_per_image = math.ceil(images_to_generate / current_no_fire)
print(f"Imágenes originales en No_Fire: {current_no_fire}")
print(f"Objetivo: {target_no_fire} imágenes.")
print(f"Generando aproximadamente {augmentations_per_image} augmentaciones por imagen.")

generated = 0
# Itera sobre cada imagen original
for img_name in os.listdir(no_fire_dir):
    img_path = os.path.join(no_fire_dir, img_name)
    try:
        # Cargar la imagen
        img = load_img(img_path)
    except Exception as e:
        print(f"Error al cargar {img_path}: {e}")
        continue

    # Convertir la imagen a array y darle la forma correcta para el generador
    img_array = img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)

    count = 0
    # Genera un número fijo de imágenes augmentadas para esta imagen original
    for batch in datagen.flow(img_array, batch_size=1,
                              save_to_dir=output_dir,
                              save_prefix='aug',
                              save_format='jpeg'):
        count += 1
        generated += 1
        print(f"Generadas {generated} imagen(es) a partir de {img_name}")
        if count >= augmentations_per_image:
            break  # Pasa a la siguiente imagen original

print(f"Aumentación completada. Nuevas imágenes generadas: {generated}")
