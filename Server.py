import socket
import pickle
import cv2
import numpy as np

# Konfiguracja serwera
HOST = "127.0.0.1"  # Nasłuch na wszystkich interfejsach sieciowych
PORT = 5000
HEADERSIZE = 10

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
        full_msg = b''
        isfull_msg = False
        new_msg = True

        while not isfull_msg:
            data = conn.recv(10000)
            if new_msg:
                print("new msg len:",data[:HEADERSIZE])
                msglen = int(data[:HEADERSIZE])
                new_msg = False
                print(f"full message length: {msglen}")
            full_msg += data
            if len(full_msg)-HEADERSIZE == msglen:
                print("full msg recvd")
                print(full_msg[HEADERSIZE:])
                print(pickle.loads(full_msg[HEADERSIZE:]))
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