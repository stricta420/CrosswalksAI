import socket
import pickle
import cv2
import numpy as np

# Konfiguracja serwera
HOST = "127.0.0.1"  # Nasłuch na wszystkich interfejsach sieciowych
PORT = 5000

def analyze_image(image):
    # Prosta analiza obrazu - wykrywanie krawędzi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Serwer nasłuchuje na {HOST}:{PORT}...")

    conn, addr = server_socket.accept()
    print(f"Połączono z {addr}")

    while True:
        # Odbiór obrazu
        data = conn.recv(4096)
        if not data:
            break

        # Deserializacja obrazu
        frame_data = pickle.loads(data)
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

        # Analiza obrazu
        analyzed_frame = analyze_image(frame)

        # Serializacja wyników analizy
        _, buffer = cv2.imencode('.jpg', analyzed_frame)
        analyzed_data = pickle.dumps(buffer)

        # Wysyłanie wyników analizy
        conn.sendall(analyzed_data)

    conn.close()
    server_socket.close()

if __name__ == "__main__":
    main()