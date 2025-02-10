import os

def count_files(directory):
    try:
        return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    except FileNotFoundError:
        print("Error: La ruta especificada no existe.")
        return -1
    except PermissionError:
        print("Error: No tienes permisos para acceder a esta ruta.")
        return -1

if __name__ == "__main__":
    # path = input("Ingrese la ruta del directorio: ")
    path = "data/processed/Training/No_Fire"
    file_count = count_files("data/processed/Training/No_Fire")
    if file_count != -1:
        print(f"Número de archivos en '{path}': {file_count}")
