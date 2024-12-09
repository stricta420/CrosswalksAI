import torch
import cv2
import os
from ultralytics import YOLOv10
from deepface import DeepFace

class ImageInfo:
    def __init__(self, image, label, confidence):
        self.image = image
        self.label = label
        self.confidence = confidence

    def print(self):
        print(self.label)
        print(self.confidence)


class AgeModel:


    def __init__(self):
        self.model_path = '../models/human.pt'
        self.yolo_model = YOLOv10(self.model_path)

    #Breaks down image into few images containing 1 person each
    #returns array of those images
    def break_down_image(self, image_path):
        image = cv2.imread(image_path)
        results = self.yolo_model(image)
        boxes = results[0].boxes  # Obiekt zawierający bounding boxy
        names = results[0].names  # Słownik z nazwami klas
        all_results = []
        for i, box in enumerate(boxes):
            # 'box' jest obiektem klasy Boxes, więc musimy uzyskać współrzędne
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Pobierz współrzędne jako ndarray
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Konwertujemy na int

            # Pobierz etykietę i pewność detekcji
            label = names[int(box.cls)]  # Etykieta (klasa obiektu)
            confidence = box.conf.item()  # Pewność detekcji

            # Wycinanie obrazu z pojedynczym obiektem
            cropped_object = image[y1:y2, x1:x2]
            new_img = ImageInfo(cropped_object, label, confidence)
            all_results.append(new_img)
            # Zapisz wykryty obiekt do pliku
            # output_file = os.path.join(output_folder,
            #                            f'detected_object_{i + 1}_{label}_{confidence:.2f}.jpg')  # Nazwa pliku
            # cv2.imwrite(output_file, cropped_object)  # Zapisz obraz
        return all_results

    def get_ages_from_images(self, images):
        ages = []
        for image_single in images:
            analize = DeepFace.analyze(img_path=image_single.image, actions=['age'], enforce_detection=False) # z jakiegoś powodu enforce_detection = False powoduje że błąd się nie pojawia i analizowane są WSZYSTKIE twarze (gdzie bez tego wywala błąd bo 1 nie może zanalizować) dlaczego? nie wiem, powinno być inaczej
            ages.append(analize[0]['age'])
        return ages

    def categorise_ages(self, ages):
        categorie = {
            "-2" : 0, #dziecko nie chodzi samodzielnie
            "3-5" : 0,
            "6-9": 0,
            "10-13": 0,
            "14-18": 0,
            "19-50": 0,
            "51-60": 0,
            "61-70": 0,
            "71-75": 0,
            "76-80": 0,
            "80+": 0
        }

        for age in ages:
            if age < 3:
                categorie["-3"] += 1
            elif age <= 5:
                categorie["3-5"] += 1
            elif age < 10:
                categorie["6-9"] += 1
            elif age < 14:
                categorie["10-13"] += 1
            elif age < 19:
                categorie["14-18"] += 1
            elif age < 51:
                categorie["19-50"] += 1
            elif age < 61:
                categorie["51-60"] += 1
            elif age < 71:
                categorie["61-70"] += 1
            elif age < 76:
                categorie["71-75"] += 1
            elif age < 81:
                categorie["76-80"] += 1

        return categorie

    def get_categories_from_photo(self, path):
        braked_images = self.break_down_image(path)
        ages = self.get_ages_from_images(braked_images)
        categorys = self.categorise_ages(ages)
        return categorys

age_model = AgeModel()
print(age_model.get_categories_from_photo("test2.jpg"))
# res = age_model.break_down_image("test2.jpg")
# for img in res:
#     analiza = DeepFace.analyze(img_path=img.image, actions=['age'], enforce_detection=False)
#     print("Predykcja wieku:", analiza[0]['age'])
#
