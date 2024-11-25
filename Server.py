import socket
import pickle
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLOv10

# Konfiguracja serwera
HOST = "127.0.0.1"  # Nasłuch na wszystkich interfejsach sieciowych
PORT = 5000
HEADERSIZE = 10
MODEL_PATH = r'C:\\Users\\Stasiu\\Desktop\\crosswalks\\CrosswalksAI\\models\\human.pt'
MODEL_PATH1 = r'C:\\Users\\Stasiu\\Desktop\\crosswalks\\CrosswalksAI\\models\\zebra.pt'
MODEL_PATH2 = r'C:\\Users\\Stasiu\\Desktop\\crosswalks\\CrosswalksAI\\models\\best.pt'

models = []
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

def appendModel(modelPath):
        models.append(YOLOv10(modelPath))

def analyze_image(image):
    Rs = []
    for model in models:
        Rs.append(model(image)[0])
    
    for results in Rs:
        detections = sv.Detections.from_ultralytics(results)
        annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    return annotated_image


def main():
    appendModel(MODEL_PATH)
    appendModel(MODEL_PATH1)
    appendModel(MODEL_PATH2)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Serwer nasłuchuje na {HOST}:{PORT}...")

    conn, addr = server_socket.accept()
    print(f"Połączono z {addr}")

    while True:
        # Odbiór obrazu
        full_msg = b''
        isfull_msg = False
        new_msg = True

        while not isfull_msg:
            data = conn.recv(4096)
            if new_msg:
                print("new msg len:",data[:HEADERSIZE])
                msglen = int(data[:HEADERSIZE])
                new_msg = False
                print(f"full message length: {msglen}")
            full_msg += data
            if len(full_msg)-HEADERSIZE == msglen:
               
                isfull_msg = True
        
        
        # Deserializacja obrazu
        frame_data = pickle.loads(full_msg[HEADERSIZE:])
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

        # Analiza obrazu
        analyzed_frame = analyze_image(frame)

        # Serializacja wyników analizy
        _, buffer = cv2.imencode('.jpg', analyzed_frame)
        analyzed_data = pickle.dumps(buffer)
        analyzed_data = bytes(f"{len(analyzed_data):<{HEADERSIZE}}", 'utf-8') + analyzed_data

        # Wysyłanie wyników analizy
        conn.send(analyzed_data)
        full_msg = b""
        new_msg = True

    conn.close()
    server_socket.close()

if __name__ == "__main__":
    main()