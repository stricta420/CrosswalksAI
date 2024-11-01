import torch
import torchvision.transforms as transforms
import VideoHandling as vh
import requests
import json

# Konfiguracja serwera
server_url = "http://127.0.0.1:5000/process_tensor"

# Konwersja obrazu na tensor i serializacja do listy


def image_to_tensor(frame):
    #image = Image.open(image_path)
    transform = transforms.ToTensor()  # Konwersja do tensora
    tensor = vh.tensorConvert(frame) #Mozliwe tensorBWConvert dla czarnobialych zdj
    tensor_list = tensor.tolist()  # Konwersja tensora na listę
    return tensor_list


# Przesyłanie tensora w obiekcie JSON
def send_tensor_to_server(tensor_list):
    data = {
        "metadata": {
            "client_id": "client_123",
            "description": "Tensor z obrazu"
        },
        "tensor_data": tensor_list
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(server_url, data=json.dumps(data), headers=headers)
    print("Odpowiedź serwera:", response.json())
    #print(data)

# Testowe działanie
send_tensor_to_server(vh.process_image_jak_u_stasia_tylko_inaczej(vh.takeAFakePicture("root/label/valid/trojkat/trojkat_1.png")).tolist())
send_tensor_to_server(vh.process_image_jak_u_stasia_tylko_inaczej(vh.takeAFakePicture("root/label/valid/trojkat/trojkat_4.png")).tolist())
send_tensor_to_server(vh.process_image_jak_u_stasia_tylko_inaczej(vh.takeAFakePicture("root/label/valid/kwadrat/kwadrat_2.png")).tolist())
send_tensor_to_server(vh.process_image_jak_u_stasia_tylko_inaczej(vh.takeAFakePicture("root/label/valid/kolo/kolo_2.png")).tolist())
send_tensor_to_server(vh.process_image_jak_u_stasia_tylko_inaczej(vh.takeAFakePicture("root/label/valid/trojkat/trojkat_5.png")).tolist())
send_tensor_to_server(vh.process_image_jak_u_stasia_tylko_inaczej(vh.takeAFakePicture("root/label/valid/trojkat/trojkat_7.png")).tolist())
send_tensor_to_server(vh.process_image_jak_u_stasia_tylko_inaczej(vh.takeAFakePicture("root/label/valid/kwadrat/kwadrat_1.png")).tolist())
send_tensor_to_server(vh.process_image_jak_u_stasia_tylko_inaczej(vh.takeAFakePicture("root/label/valid/kolo/kolo_3.png")).tolist())
send_tensor_to_server(vh.process_image_jak_u_stasia_tylko_inaczej(vh.takeAFakePicture("root/label/valid/trojkat/trojkat_11.png")).tolist())
send_tensor_to_server(vh.process_image_jak_u_stasia_tylko_inaczej(vh.takeAFakePicture("root/label/valid/trojkat/trojkat_14.png")).tolist())
send_tensor_to_server(vh.process_image_jak_u_stasia_tylko_inaczej(vh.takeAFakePicture("root/label/valid/kwadrat/kwadrat_12.png")).tolist())
send_tensor_to_server(vh.process_image_jak_u_stasia_tylko_inaczej(vh.takeAFakePicture("root/label/valid/kolo/kolo_12.png")).tolist())