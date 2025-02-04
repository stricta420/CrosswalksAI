import cv2
import socket
import pickle
import time
import numpy as np
import os
from pathlib import Path
import ssl

HOME = Path(__file__).resolve().parent.parent

class VideoClient:
    def __init__(self, server_ip, port, headersize=10):
        self.server_ip = server_ip
        self.port = port
        self.headersize = headersize
        self.home = Path(__file__).resolve().parent.parent
        self.client_socket = None
        self.cap = None
        self.signal = True
        self.max_time = []

    def connect_to_server(self):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            self.context = ssl.create_default_context()
            self.context.check_hostname = False
            self.context.verify_mode = ssl.CERT_NONE #wyłączona weryfikacja dla testów
            self.client_socket = self.context.wrap_socket(self.client_socket, server_hostname=self.server_ip)

            self.client_socket.connect((self.server_ip, self.port))

            print(f"Połączono z serwerem {self.server_ip}:{self.port}")
        except Exception as e:
            print(f"Błąd podczas łączenia z serwerem: {e}")
            raise

    def open_video(self, video_path):
        self.cap = cv2.VideoCapture(self.home / video_path)
        if not self.cap.isOpened():
            raise Exception("Nie można otworzyć pliku wideo")

    def send_frame(self, frame):
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = pickle.dumps(buffer)
            frame_data = bytes(f"{len(frame_data):<{self.headersize}}", 'utf-8') + frame_data
            self.client_socket.send(frame_data)
        except Exception as e:
            print(f"Błąd podczas wysyłania klatki: {e}")
            raise

    def receive_analyzed_frame(self):
        full_msg = b''
        new_msg = True

        while True:
            data = self.client_socket.recv(4096)
            if new_msg:
                if len(data) < self.headersize:
                    raise Exception("Niepełne dane nagłówka, przerywam.")
                msglen = int(data[:self.headersize])
                new_msg = False

            full_msg += data

            if len(full_msg) - self.headersize == msglen:
                analyzed_frame_data = pickle.loads(full_msg[self.headersize:])
                return cv2.imdecode(np.frombuffer(analyzed_frame_data, np.uint8), cv2.IMREAD_COLOR)


    def receive_message(self):
        full_msg = b''
        new_msg = True

        while True:
            data = self.client_socket.recv(4096)
            if new_msg:
                if len(data) < self.headersize:
                    raise Exception("Niepełne dane nagłówka, przerywam.")
                msglen = int(data[:self.headersize])
                new_msg = False

            full_msg += data

            if len(full_msg) - self.headersize == msglen:
                return pickle.loads(full_msg[self.headersize:])
                

    def run(self, video_path):
        try:
            self.connect_to_server()
            self.open_video(video_path)

            # Pobierz FPS wideo
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            frames_to_skip = int(fps * 0.5)  # Liczba klatek do pominięcia, aby wysyłać co 0.5 sekundy

            frame_counter = 0  # Licznik klatek
            while True:
                counter = 0
                while True:
                    
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Nie można odczytać klatki z pliku wideo")
                        break

                    # Wysyłaj tylko co frames_to_skip klatkę
                    if self.signal and frame_counter % frames_to_skip == 0:
                        self.send_frame(frame)  # Wysyłanie klatki
                        message = self.receive_message()  # Odbieranie odpowiedzi
                        print(f"Zbieranie danych podczas czerwonego: {message}")
                        self.max_time.append(message)
                        counter=counter+1
                    elif counter==5: 
                        self.signal = False 
                        break
                    frame_counter += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if counter==5:
                    self.signal = True
                    maxt = max(self.max_time)
                    print(f"Zielone: {maxt}")
                    time.sleep(maxt)
                    counter=0
                    self.max_time = []
        except Exception as e:
            print(f"Błąd w trakcie działania: {e}")
        finally:
            self.cleanup()


    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.client_socket:
            self.client_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    SERVER_IP = '127.0.0.1'  # Publiczny adres IP serwera
    PORT = 5000
    VIDEO_PATH = HOME/"video/wideo4.mp4"


    client = VideoClient(SERVER_IP, PORT)
    client.run(VIDEO_PATH)
