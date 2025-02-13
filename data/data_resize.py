from tensorflow.keras.preprocessing.image import load_img, save_img
import os
from PIL import Image

# Función para redimensionar imágenes
def resize_images(input_dir, output_dir, target_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)
        
        for img_name in os.listdir(category_path):
            # Filtrar archivos por extensión
            if not img_name.lower().endswith(valid_extensions):
                continue
            
            img_path = os.path.join(category_path, img_name)
            try:
                # Cargar la imagen
                img = load_img(img_path)
                # Redimensionar usando LANCZOS (compatible con Pillow 8)
                img_resized = img.resize(target_size, Image.LANCZOS)
                # Guardar la imagen redimensionada
                output_path = os.path.join(output_category_path, img_name)
                img_resized.save(output_path)
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
                continue
                
    print(f"Redimensionamiento completado en: {output_dir}")

# Rutas (ajusta según la estructura de tu proyecto)
train_dir = "data/raw/Training"
test_dir  = "data/raw/Test"

train_resized_dir = "data/processed/Training"
test_resized_dir = "data/processed/Test"

# Redimensionar imágenes
resize_images(train_dir, train_resized_dir)
resize_images(test_dir, test_resized_dir)
