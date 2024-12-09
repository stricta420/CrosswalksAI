import torch
import cv2
import os
from ultralytics import YOLOv10
from deepface import DeepFace
import math
import copy

class ImageInfo:
    def __init__(self, image, label, confidence, x1=None, y1=None, x2=None, y2=None):
        self.image = image
        self.label = label
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def print(self):
        print(self.label)
        print(self.confidence)

class GodModel:
    def __init__(self):
        self.human_model = None
        self.disable_model = None
        self.zebra_model = None
        self.set_up_models()

    def set_up_models(self, human='../models/human.pt', disable='../models/best.pt', zebra='../models/zebra.pt'):
        self.human_model = YOLOv10(human)
        self.disable_model = YOLOv10(disable)
        self.zebra_model = YOLOv10(zebra)

    def disabilyty_scan(self, frame):
        resoult = self.disable_model(frame)
        boxes = resoult[0].boxes
        names = resoult[0].names
        all_results = []
        for i, box in enumerate(boxes):
            # 'box' jest obiektem klasy Boxes, więc musimy uzyskać współrzędne
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Pobierz współrzędne jako ndarray
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Konwertujemy na int

            # Pobierz etykietę i pewność detekcji
            label = names[int(box.cls)]  # Etykieta (klasa obiektu)
            confidence = box.conf.item()  # Pewność detekcji

            # Wycinanie obrazu z pojedynczym obiektem
            cropped_object = frame[y1:y2, x1:x2]
            new_img = ImageInfo(cropped_object, label, confidence, x1, y1, x2, y2)
            all_results.append(new_img)
        return all_results

    def break_down_image(self, frame):
        results = self.human_model(frame)
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
            cropped_object = frame[y1:y2, x1:x2]
            new_img = ImageInfo(cropped_object, label, confidence, x1, y1, x2, y2)
            all_results.append(new_img)
            # Zapisz wykryty obiekt do pliku
            # output_file = os.path.join(output_folder,
            #                            f'detected_object_{i + 1}_{label}_{confidence:.2f}.jpg')  # Nazwa pliku
            # cv2.imwrite(output_file, cropped_object)  # Zapisz obraz
        return all_results

    def get_ages_from_images(self, images):
        ages = []
        for image_single in images:
            analize = DeepFace.analyze(img_path=image_single.image, actions=['age'],
                                       enforce_detection=False)  # z jakiegoś powodu enforce_detection = False powoduje że błąd się nie pojawia i analizowane są WSZYSTKIE twarze (gdzie bez tego wywala błąd bo 1 nie może zanalizować) dlaczego? nie wiem, powinno być inaczej
            ages.append(analize[0]['age'])
        return ages

    #Odleglosc euklidesowa srodkow prostokatow
    def get_distance_from_bounding_boxes_of_images(self, image1, image2):
        x1_1 = image1.x1
        x2_1 = image1.x2
        y1_1 = image1.y1
        y2_1 = image1.y2

        center_1x = (x1_1 + x2_1) // 2
        center_1y = (y1_1 + y2_1) // 2

        x1_2 = image2.x1
        x2_2 = image2.x2
        y1_2 = image2.y1
        y2_2 = image2.y2

        center_2x = (x1_2 + x2_2) // 2
        center_2y = (y1_2 + y2_2) // 2

        distance = math.sqrt(math.pow(center_1x - center_2x, 2) + math.pow(center_1y - center_2y, 2))
        return distance



    def find_closest_box(self, image, all_images):
        lowest_distane = None
        closest_img_ind = None
        for i in range(len(all_images)):
            distance = self.get_distance_from_bounding_boxes_of_images(image, all_images[i])
            if lowest_distane is None:
                lowest_distane = distance
                closest_img_ind = i
                continue

            if lowest_distane < distance:
                lowest_distane = distance
                closest_img_ind = i
        return closest_img_ind


    #checks if object image1 is in object image2
    #we asume that camera is in a good position -
    #so ground is lower than the sky
    def is_on(self, image1, image2):
        #pozycja lewej stopy
        low_x = image1.x2
        low_y = image1.y2

        if image2.x2 > low_x > image2.x1 and  image2.y2 > low_y > image2.y1:
            return True
        else:
            return False

    def ignore_people_on_crosswalk(self, crosswalk_img, people):
        new_people = []
        for person in people:
            if not self.is_on(person, crosswalk_img):
                new_people.append(copy.deepcopy(person))
        return new_people

    #TO DO: add logic to exclude people that are already on zebra corosing
    #returns : number_of_people, maks_age,
    def analize_frame(self, frame):
        people = self.break_down_image(frame)
        ages = self.get_ages_from_images(people)
        zebra = self.zebra_model(frame)
        #people_not_on_zebra = self.ignore_people_on_crosswalk(zebra, people)
        disabilytis = self.disabilyty_scan(frame)
        disble_with_age = {}
        if len(disabilytis) != 0:
            for i in range(len(disabilytis)):
                dis = disabilytis[i]
                closest_indeks = self.find_closest_box(dis, people)
                if dis.label not in disble_with_age:
                    disble_with_age[dis.label] = [ages[closest_indeks]]
                else:
                    disble_with_age[dis.label] += [ages[closest_indeks]] #tablica wewnątrz słownika - stąd dodawanie [ages[i]] zamiast ages[i]
        return disble_with_age, ages



god_mod = GodModel()
image_path = "test2.jpg"
frame = cv2.imread(image_path)
print(god_mod.analize_frame(frame))
