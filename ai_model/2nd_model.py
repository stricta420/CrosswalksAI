from training import Model
import torch
import cv2
import numpy as np
from DataLoaders import DataLoaders
import torch
import os
import numpy as np
from PIL import Image
from pathlib import Path
import cv2


class ModelYolo(Model):
    def __init__(self):
        self.model = None  # Model YOLOv7 będzie ładowany osobno
        self.dataModules = DataLoaders()  # Zakładamy, że DataLoaders obsługuje YOLOv7-style dane
        self.path_to_model = "../../yolov7/yolov7.pt"  # Ścieżka do modelu YOLOv7
        self.labels_map = None  # Słownik klas, przypisuje ID do etykiet
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.configure_model()
        super().__init__()
        self.path_to_model = "../../yolov7/yolov7.pt"

    def configure_model(self):
        """
        Konfiguruje model YOLOv7, ładując go lokalnie z podanej ścieżki.
        """
        print("Ładowanie modelu YOLOv7 z lokalnego pliku...")
        print(f"Ścieżka do modelu: {self.path_to_model}")

        # Wczytywanie modelu zapisanego jako pełny obiekt (z torch.save(model))
        self.model = torch.load(self.path_to_model, map_location=self.device)
        print("mo0del wczytany z zapisu")
        # Przeniesienie modelu na odpowiednie urządzenie (CPU lub GPU)
        self.model.to(self.device)

        # Ustawienie trybu ewaluacyjnego
        self.model.eval()

        print("Model YOLOv7 został załadowany i skonfigurowany.")

    def process_image(self, image_path):
        """
        Przetwarza obraz na tensor zgodny z wymogami YOLOv7.
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))  # Rozmiar zgodny z YOLOv7
        img = img / 255.0  # Normalizacja
        img = np.transpose(img, (2, 0, 1))  # Zamiana wymiarów na (C, H, W)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)
        return img

    def predict(self, image):
        """
        Przewiduje obiekty w przetworzonym obrazie.
        """
        self.model.eval()
        with torch.no_grad():
            results = self.model(image)
        return results

    def predict_from_path(self, path):
        """
        Przewiduje obiekty w obrazie na podstawie ścieżki.
        """
        image = self.process_image(path)
        results = self.predict(image)

        # Eksport wyników
        detected_objects = []
        for det in results.xyxy[0]:  # Format: x1, y1, x2, y2, confidence, class
            x1, y1, x2, y2, conf, cls = det.tolist()
            label = self.labels_map[int(cls)] if self.labels_map else f"Class {int(cls)}"
            detected_objects.append({
                "label": label,
                "bbox": (x1, y1, x2, y2),
                "confidence": conf
            })

        return detected_objects

    def train(self):
        """
        Implementacja trenowania YOLOv7.
        YOLOv7 wymaga dodatkowego skryptu treningowego.
        """
        print("Trenowanie YOLOv7 nie jest zaimplementowane w tej klasie.")
        # YOLOv7 wymaga własnego skryptu treningowego - można wywołać zewnętrzny proces
        pass

    def save(self, path=None):
        """
        Zapisuje wytrenowany model YOLOv7.
        """
        if not path:
            path = self.path_to_model
        torch.save(self.model.state_dict(), path)
        print(f"Model YOLOv7 zapisano w {path}")

    def load_eval(self):
        """
        Ładuje model YOLOv7 w trybie ewaluacyjnym.
        """
        base_dir = Path(__file__).resolve().parent
        full_path = base_dir / self.path_to_model
        if not full_path.exists():
            raise FileNotFoundError(f"Model nie istnieje w {full_path}")
        self.model.load_state_dict(torch.load(full_path, map_location=self.device))
        self.model.eval()
        print("Model YOLOv7 załadowany w trybie ewaluacyjnym.")

# Przykładowe użycie
if __name__ == "__main__":
    # Inicjalizacja modelu YOLOv7
    ai = ModelYolo()

    # Przypisanie mapy etykiet (przykład)
    ai.set_labels_map({
        0: "Person",
        1: "Bicycle",
        2: "Car",
        3: "Dog",
    })
    ai.configure_model()
    # Załaduj model w trybie ewaluacyjnym
    ai.load_eval()

    # Przewidywanie obiektów w obrazie
    image_path = "1_jpg.rf.a69b5ab2d21edcf423766db6ee991165.jpg"  # Ścieżka do obrazu
    results = ai.predict_from_path(image_path)

    # Wyświetlanie wyników
    for obj in results:
        print(f"Label: {obj['label']}, BBox: {obj['bbox']}, Confidence: {obj['confidence']:.2f}")
