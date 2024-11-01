import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

class DataLoaders:

    def __init__(self):
        self.transformations = self.get_transofrmation()
        self.train_loader, self.val_loader = self.get_data_loaders()

    def get_transofrmation(self, resize=255, center_crop=224,normalize_mean=None,normalize_std=None):
        if normalize_mean is None:
            normalize_mean = [0.485, 0.456, 0.406]
        if normalize_std is None:
            normalize_std = [0.229, 0.224, 0.225]
        transformations = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(center_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        return transformations

    def get_data_loaders(self, train_set_folder="root\\label\\train", val_set_folder="root\\label\\valid"):
        # Ustalanie pełnych ścieżek względem bieżącego folderu projektu
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Katalog, w którym znajduje się obecny plik
        train_set_path = os.path.join(base_dir, train_set_folder)
        val_set_path = os.path.join(base_dir, val_set_folder)

        # Sprawdzenie, czy ścieżki istnieją
        if not os.path.isdir(train_set_path) or not os.path.isdir(val_set_path):
            raise FileNotFoundError(f"Nie znaleziono ścieżki: {train_set_path} lub {val_set_path}")

        # Wczytanie zbiorów danych i zastosowanie transformacji
        train_set = datasets.ImageFolder(train_set_path, transform=self.transformations)
        val_set = datasets.ImageFolder(val_set_path, transform=self.transformations)

        # Umieszczenie danych w DataLoader
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)
        return train_loader, val_loader