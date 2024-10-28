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

def takeAPicture():
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cap.release()
        return frame
    else:
        cap.release()
        raise Exception("Nie udało się zrobić zdjęcia.")
    
def takeAFakePicture(path):
    frame = cv.imread(path)
    if frame is None:
        raise Exception("Nie udało się zrobić zdjęcia.")
    else:
        return frame

#funkcja przeksztalcajaca obraz na podobnie do funkcji stasia, ale tylko uzywajac opencv
def process_image_jak_u_stasia_tylko_inaczej(img):  
    # Zmiana rozmiaru: dostosowanie krótszego boku do 255 pikseli, zachowanie proporcji
    height, width = img.shape[:2]
    if width < height:
        new_width = 255
        new_height = int(255 * height / width)
    else:
        new_height = 255
        new_width = int(255 * width / height)
    img = cv.resize(img, (new_width, new_height))
    
    # Przycinanie do rozmiaru 224x224
    height, width = img.shape[:2]
    left = (width - 224) // 2
    top = (height - 224) // 2
    img = img[top:top+224, left:left+224]
    
    # Konwersja obrazu z BGR na RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Normalizacja pikseli (skala 0-1)
    img = img / 255.0

    # Normalizacja wartości dla kanałów R, G i B
    img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229  # R
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224  # G
    img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225  # B

    # Przekształcenie obrazu na tensor PyTorch
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    return img
#Kamera (0 - wbudowana)
#cap = cv.VideoCapture(0)

# if torch.cuda.is_available():
#     print("CUDA jest dostępne!")
# else:
#     print("CUDA nie jest dostępne.")

#inputFile('Test_Video.mp4',1)
#inputCamera()