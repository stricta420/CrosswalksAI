import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # Dodaj ten import!


# Sprawdzenie, czy GPU jest dostępne
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **1. Przygotowanie danych z ImageFolder**
transformations = transforms.Compose([
    transforms.Resize((64, 64)),   # Zmiana rozmiaru obrazów na 64x64
    transforms.ToTensor(),         # Konwersja na tensor
    transforms.Normalize([0.5], [0.5])  # Normalizacja do przedziału [-1, 1]
])

train_set = datasets.ImageFolder("root/label/train", transform=transformations)
val_set = datasets.ImageFolder("root/label/valid", transform=transformations)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)

# **2. Definicja modelu CNN**
class ShapeClassifier(nn.Module):
    def __init__(self):
        super(ShapeClassifier, self).__init__()
        # Warstwy konwolucyjne
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Warstwy normalizacji i pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

        # Warstwy w pełni połączone
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 64 mapy cech po 8x8
        self.fc2 = nn.Linear(128, 3)  # 3 klasy (kwadrat, koło, trójkąt)

    def forward(self, x):
        # Przepływ przez warstwy konwolucyjne
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # Spłaszczenie wejścia
        x = x.view(-1, 64 * 8 * 8)

        # Przepływ przez warstwy w pełni połączone
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# **3. Inicjalizacja modelu, funkcji straty i optymalizatora**
model = ShapeClassifier().to(device)
criterion = nn.CrossEntropyLoss()  # Funkcja straty
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optymalizator Adam

# **4. Trening modelu**
num_epochs = 100

for epoch in range(num_epochs):
    model.train()  # Ustawienie modelu w tryb treningowy
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass i optymalizacja
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# **5. Ewaluacja modelu**
model.eval()  # Ustawienie modelu w tryb ewaluacji
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

# **6. Przykładowe przewidywanie**
def predict_image(image_path):
    image = Image.open(image_path)
    image = transformations(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    class_names = train_set.classes
    return class_names[predicted.item()]

# Testowanie na jednym obrazie
sample_image = "root/label/valid/trojkat/trojkat_1.png"
prediction = predict_image(sample_image)
print(f"Przewidziana figura: {prediction}")

