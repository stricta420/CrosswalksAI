import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
import numpy as np

def tensorConvert(frame, target_size=(224, 224)): # Konwersja na tensor, dla RGB
    resized_frame = cv.resize(frame, target_size, interpolation=cv.INTER_LINEAR)
    frame_rgb = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB) #do rgb
    #frame_rgb = frame_rgb / 255.0 #normalizacja na przedział 0-1
    #frame_transposed = np.transpose(frame_rgb, (2, 0, 1))
    #input_data = torch.tensor(frame_transposed, dtype=torch.float32).unsqueeze(0)
    transform = transforms.ToTensor()
    input_data = transform(frame_rgb)
    return input_data

def tensorBWConvert(frame, target_size=(224, 224)): # Konwersja na tensor dla Czarno-białych klatek
    resized_frame = cv.resize(frame, target_size, interpolation=cv.INTER_LINEAR)
    gray_frame = cv.cvtColor(resized_frame,cv.COLOR_BGR2GRAY)
    #gray_frame_tensor = torch.tensor(gray_frame, dtype=torch.float32) / 255.0
    transform = transforms.ToTensor()
    gray_frame_tensor = transform(gray_frame)
    return gray_frame_tensor

def inputFile(path, interval = 1): #funkcja naśladująca import z kamery dla pliku, interwały domyślnie 1 sekundowe, aby nie robić zbyt czestych operacji

    if path == 0:
        print("Imput from Real Camera, use inputCamera() instead")
        return
    
    cap = cv.VideoCapture(path)
    fps = cap.get(cv.CAP_PROP_FPS)

    frame_duration = 1 / fps
    frame_leap = int(interval / frame_duration)
    print("Frame = " + str(frame_duration) + "s -> making leaps of " + str(frame_leap) + " frames")

    frame_number = 1

    while cap.isOpened():

        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        
        if not ret:
            print("End of Video")
            break

        cv.imshow('Frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        print(tensorBWConvert(frame))

        frame_number += frame_leap
    
    cap.release()
    cv.destroyAllWindows()

def inputCamera(interval = 1):
    cap = cv.VideoCapture(0)

    while True:
        # Odczyt jednej klatki z kamery
        ret, frame = cap.read()
        
        if not ret:
            print("Błąd odczytu z kamery")
            break

        
        cv.imshow('Kamera', frame) # <- poglądy
        #cv.imshow('B&W', cv.cvtColor(frame, cv.COLOR_BGR2GRAY)) # <- podglądy

        #Test
        #resized_frame = cv.resize(frame, (224,224), interpolation=cv.INTER_LINEAR)
        #cv.imshow('Pixelated', resized_frame)

        print(tensorBWConvert(frame)) # Czarno-biały na tensor, bo chyba nie ma sensu robic kolorowego rozpoznawania
        # Zamknij okno po naciśnięciu klawisza 'q'
        if cv.waitKey(interval*1000) & 0xFF == ord('q'):
            break


    # Zwalniamy zasoby kamery i zamykamy okno
    cap.release()
    cv.destroyAllWindows()

#Kamera (0 - wbudowana)
#cap = cv.VideoCapture(0)

if torch.cuda.is_available():
    print("CUDA jest dostępne!")
else:
    print("CUDA nie jest dostępne.")

#inputFile('Test_Video.mp4',1)
inputCamera()