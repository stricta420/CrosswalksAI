import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import random

# Parametry danych
image_size = 64  # Obraz o wymiarach 64x64
num_classes = 3  # Kwadrat, koło, trójkąt

# Generowanie syntetycznych danych: kwadrat, koło, trójkąt
def generate_shape(shape_type):
    img = Image.new('RGB', (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if shape_type == 0:  # Kwadrat
        draw.rectangle([15, 15, 50, 50], outline="black", fill="black")
    elif shape_type == 1:  # Koło
        draw.ellipse([15, 15, 50, 50], outline="black", fill="black")
    elif shape_type == 2:  # Trójkąt
        draw.polygon([ (32, 5), (5, 55), (59, 55) ], outline="black", fill="black")

    return img

# Funkcja do konwersji obrazu na tensor
def image_to_tensor(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Skala szarości
        transforms.ToTensor(),  # Konwersja na tensor
    ])
    return transform(image)

# Generowanie zestawu danych (n = liczba próbek)
def generate_dataset(n):
    images = []
    labels = []
    for _ in range(n):
        label = random.randint(0, 2)  # Losowanie etykiety (0=kwadrat, 1=koło, 2=trójkąt)
        img = generate_shape(label)  # Generowanie odpowiedniego obrazu
        img_tensor = image_to_tensor(img)
        images.append(img_tensor)
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)

# Prosty model sieci neuronowej
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(image_size * image_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(-1, image_size * image_size)  # Spłaszczenie obrazów 64x64 na wektor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Inicjalizacja modelu, funkcji straty i optymalizatora
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generowanie danych
train_images, train_labels = generate_dataset(1000)  # 1000 próbek do treningu
test_images, test_labels = generate_dataset(200)     # 200 próbek do testowania

# Trening modelu
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(0, len(train_images), batch_size):
        inputs = train_images[i:i + batch_size]
        labels = train_labels[i:i + batch_size]

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass i optymalizacja
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_images)}")

# Ewaluacja modelu
correct = 0
total = 0

with torch.no_grad():
    outputs = model(test_images)
    _, predicted = torch.max(outputs, 1)
    total = test_labels.size(0)
    correct = (predicted == test_labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")

