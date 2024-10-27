# test.py
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn 

labels_map = {
    0: "circle",  
    1: "square",  
    2: "triangle"   
}

# Procesuj obrazek
def process_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    img = img.resize((255, int(255 * (height / width))) if width < height else (int(255 * (width / height)), 255))
    width, height = img.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))
    img = np.array(img)
    img = img.transpose((2, 0, 1))
    img = img / 255
    img[0] = (img[0] - 0.485) / 0.229
    img[1] = (img[1] - 0.456) / 0.224
    img[2] = (img[2] - 0.406) / 0.225
    img = img[np.newaxis, :]
    image = torch.from_numpy(img).float()
    return image

# Użyj modelu do przewidywania etykiety
def predict(image, model):
    output = model.forward(image)
    output = torch.exp(output)
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()

# Pokaż obrazek
def show_image(image):
    image = image.numpy()
    image[0] = image[0] * 0.226 + 0.445
    fig = plt.figure(figsize=(25, 4))
    plt.imshow(np.transpose(image[0], (1, 2, 0)))

#Po prostu funkcja z calym mainem, aby do serwera wrzucić testowo
def allInOne(tensor):
    model = models.densenet161(pretrained=True)
    classifier_input = model.classifier.in_features
    num_labels = 3
    classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                            nn.ReLU(),
                            nn.Linear(1024, 512),
                            nn.ReLU(),
                            nn.Linear(512, num_labels),
                            nn.LogSoftmax(dim=1))
    model.classifier = classifier

    model.load_state_dict(torch.load('models/trained_model.pth'))
    model.eval()  # Ustaw model w trybie ewaluacji
    print("Model loaded from models/trained_model.pth")
    top_prob, top_class = predict(tensor, model)
    predicted_label = labels_map[top_class]
    return predicted_label,top_prob

# Załaduj model
model = models.densenet161(pretrained=True)
classifier_input = model.classifier.in_features
num_labels = 3
classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))
model.classifier = classifier

# Załaduj zapisany model
model.load_state_dict(torch.load('models/trained_model.pth'))
model.eval()  # Ustaw model w trybie ewaluacji
print("Model loaded from models/trained_model.pth")

# Przetwarzanie obrazka i przewidywanie
image_path = "root/label/valid/trojkat/trojkat_1.png"  # Zmień na ścieżkę do swojego obrazka testowego
image = process_image(image_path)
top_prob, top_class = predict(image, model)
show_image(image)

# Wyświetl wynik z etykietą
predicted_label = labels_map[top_class]  # Uzyskaj etykietę na podstawie klasy
print("The model is {:.2f}% certain that the image has a predicted class of {}".format(top_prob * 100, predicted_label))
