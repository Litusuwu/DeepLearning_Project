import os
import math
from PIL import Image
import torch
import torchvision.transforms as transforms

# Definir transformaciones corregidas para CNN y ViT

cnn_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomRotation(degrees=(-7, 7), fill=0),  
    transforms.ColorJitter(brightness=(0.8, 1.1), contrast=(0.8, 1.1), saturation=(0.8, 1.1)),  # 🔹 Mejora el brillo
    transforms.Resize((224, 224)),  
    transforms.RandomGrayscale(p=0.05), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

vit_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-7, 7), fill=0),  
    transforms.ColorJitter(brightness=(0.8, 1.1), contrast=(0.8, 1.1), saturation=(0.8, 1.1)),  
    transforms.Resize((224, 224)),
    transforms.RandomGrayscale(p=0.05),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

# Función para aplicar aumentación y guardar imágenes
def data_augmentation(input_dir, output_dir, transformation, target_size):
    os.makedirs(output_dir, exist_ok=True)

    # Obtener lista de imágenes en la carpeta
    images_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
    current = len(images_list)

    if current == 0:
        raise ValueError(f"No se encontraron imágenes en {input_dir}.")

    images_to_generate = target_size - current
    augmentations_per_image = images_to_generate // current  # Mejor control sobre cuántas generar
    remaining_images = images_to_generate % current  # Para distribuir si no es exacto

    print(f"Procesando {input_dir}")
    print(f"- Imágenes originales: {current}")
    print(f"- Objetivo: {target_size} imágenes")
    print(f"- Augmentaciones por imagen: {augmentations_per_image}")

    generated = 0
    unique_id = 1  # ID único para nombres de archivo

    for i, img_name in enumerate(images_list):
        img_path = os.path.join(input_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB")  # Cargar imagen en RGB
        except Exception as e:
            print(f"Error al cargar {img_path}: {e}")
            continue

        total_augmentations = augmentations_per_image + (1 if i < remaining_images else 0)  # Distribuir imágenes extra

        for _ in range(total_augmentations):
            augmented_img = transformation(img)  # Aplicar transformación
            augmented_img = transforms.functional.to_pil_image(augmented_img)  # Convertir tensor a PIL
            augmented_img.save(os.path.join(output_dir, f"aug_{unique_id}.jpg"))  # Guardar imagen aumentada
            unique_id += 1
            generated += 1
            if generated >= images_to_generate:
                break  # Salir si se alcanza el objetivo

    print(f"Aumentación completada: {generated} nuevas imágenes generadas en {output_dir}.\n")

# Definir rutas de entrada y salida
train_in_dir_fire = "resized/Training/Fire"
train_out_dir_fire_cnn = "augmented/CNN/Training/Fire"
train_out_dir_fire_vit = "augmented/VIT/Training/Fire"

train_in_dir_no_fire = "resized/Training/No_Fire"
train_out_dir_no_fire_cnn = "augmented/CNN/Training/No_Fire"
train_out_dir_no_fire_vit = "augmented/VIT/Training/No_Fire"

# Ejecutar aumentación para cada conjunto
data_augmentation(train_in_dir_fire, train_out_dir_fire_cnn, cnn_transforms, 50036)
data_augmentation(train_in_dir_fire, train_out_dir_fire_vit, vit_transforms, 50036)
data_augmentation(train_in_dir_no_fire, train_out_dir_no_fire_cnn, cnn_transforms, 28714)
data_augmentation(train_in_dir_no_fire, train_out_dir_no_fire_vit, vit_transforms, 28714)
