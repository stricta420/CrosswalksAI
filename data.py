import os
import random
from PIL import Image, ImageDraw

# Ścieżki do folderów dla zbiorów treningowych i walidacyjnych
train_dir = "root/label/train"
valid_dir = "root/label/valid"

# Nazwy klas (foldery)
classes = ["kwadrat", "kolo", "trojkat"]

# Tworzenie struktury katalogów
def create_folders():
    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)

# Funkcja do generowania losowego koloru
def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))  # (R, G, B)

# Funkcja do generowania losowej wielkości i pozycji figury
def random_shape_position():
    size = random.randint(20, 50)  # Losowy rozmiar figury
    x1, y1 = random.randint(0, 64 - size), random.randint(0, 64 - size)  # Losowe położenie
    x2, y2 = x1 + size, y1 + size
    return [(x1, y1), (x2, y2)]

# Funkcja do generowania losowego obrazu figury
def generate_shape(shape_type):
    img = Image.new('RGB', (64, 64), (255, 255, 255))  # Białe tło
    draw = ImageDraw.Draw(img)
    color = random_color()  # Losowy kolor figury

    if shape_type == "kwadrat":
        coords = random_shape_position()
        draw.rectangle(coords, outline=color, fill=color)

    elif shape_type == "kolo":
        coords = random_shape_position()
        draw.ellipse(coords, outline=color, fill=color)

    elif shape_type == "trojkat":
        # Losowy trójkąt w różnych pozycjach
        points = [
            (random.randint(5, 59), random.randint(0, 30)),
            (random.randint(0, 20), random.randint(40, 64)),
            (random.randint(40, 64), random.randint(40, 64))
        ]
        draw.polygon(points, outline=color, fill=color)

    return img

# Funkcja do generowania i zapisywania obrazów
def generate_and_save_images(num_images, dataset="train"):
    for class_name in classes:
        for i in range(num_images):
            img = generate_shape(class_name)

            # Zapisz obraz do odpowiedniego folderu
            folder = train_dir if dataset == "train" else valid_dir
            img.save(os.path.join(folder, class_name, f"{class_name}_{i + 1}.png"))

# Utwórz foldery
create_folders()

# Wygeneruj i zapisz obrazy (np. 100 do treningu, 20 do walidacji)
generate_and_save_images(100, dataset="train")
generate_and_save_images(20, dataset="valid")

print("Obrazy zostały wygenerowane i zapisane.")
