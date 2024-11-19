import cv2

from deepface import DeepFace

img = cv2.imread("group2.jpg")

results = DeepFace.analyze(img, actions=["age"], enforce_detection=False)  # Możesz ustawić `enforce_detection` na False, aby nie rzucało błędów, jeśli nie wykryje twarzy

# Wyświetl wyniki
for i, result in enumerate(results):
    age = result["age"]
    print(f"Osoba {i+1}: Wiek = {age}")