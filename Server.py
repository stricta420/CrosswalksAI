import socket
import pickle
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLOv10
from pathlib import Path
import ssl

HOME = Path(__file__).resolve().parent

class VideoServer:
    def __init__(self, host, port, certificate=None, key=None, headersize=10):
        self.host = host
        self.port = port
        self.certificate = certificate if certificate is not None else None
        self.key = key if key is not None else None
        self.headersize = headersize
        self.home = Path(__file__).resolve().parent
        self.models = []
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.server_socket = None

    def load_model(self, model_path):
        try:
            model = YOLOv10(model_path)
            self.models.append(model)
        except Exception as e:
            print(f"Błąd podczas ładowania modelu {model_path}: {e}")
    
    def secure_socket(self):
        # Konfiguracja SSL
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.load_cert_chain(certfile=self.certificate, keyfile=self.key)

        # Owijanie gniazda w SSL
        self.server_socket = self.context.wrap_socket(self.server_socket, server_side=True)
        print("Socket secured!")

    def setup_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind((self.host, self.port))

            if(self.certificate!=None and self.key!=None):
                self.secure_socket()
            else:
                print("Socket not secure!")

            self.server_socket.listen()
            print(f"Serwer nasłuchuje na {self.host}:{self.port}...")
        except Exception as e:
            print(f"Błąd podczas konfiguracji serwera: {e}")
            raise

    def analyze_image(self, image):
        try:
            for model in self.models:
                results = model(image)[0]
                detections = sv.Detections.from_ultralytics(results)
                # Sprawdzenie, czy wykryto osobę
                for label in detections.class_id:
                    if label == 0:  # Zakładając, że klasa "person" ma ID 0
                        return "zielone światło"
            return "czerwone światło"
        except Exception as e:
            print(f"Błąd podczas analizy obrazu: {e}")
            raise

    def handle_client(self, conn):
        try:
            while True:
                full_msg = b''
                new_msg = True

                while True:
                    data = conn.recv(4096)
                    if new_msg:
                        if len(data) < self.headersize:
                            raise Exception("Niepełne dane nagłówka, przerywam.")
                        msglen = int(data[:self.headersize])
                        new_msg = False

                    full_msg += data

                    if len(full_msg) - self.headersize == msglen:
                        frame_data = pickle.loads(full_msg[self.headersize:])
                        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                        analysis_result = self.analyze_image(frame)

                        # Wysyłanie wyników analizy (zielone lub czerwone światło)
                        result_data = pickle.dumps(analysis_result)
                        result_data = bytes(f"{len(result_data):<{self.headersize}}", 'utf-8') + result_data
                        conn.send(result_data)
                        break
        except Exception as e:
            print(f"Błąd podczas obsługi klienta: {e}")
        finally:
            conn.close()

    def run(self):
        self.setup_server()

        while True:
            try:
                conn, addr = self.server_socket.accept()
                print(f"Połączono z {addr}")
                self.handle_client(conn)
            except Exception as e:
                print(f"Błąd podczas akceptacji połączenia: {e}")
            
    def cleanup(self):
        if self.server_socket:
            self.server_socket.close()

if __name__ == "__main__":
    HOST = "127.0.0.1"  # Adres hosta
    PORT = 5000          # Port nasłuchiwania

    CERT_FILE = HOME/"klucze/cert.pem"
    KEY_FILE = HOME/"klucze/key.pem"

    server = VideoServer(HOST, PORT,CERT_FILE,KEY_FILE)
    server.load_model(HOME/"models/human.pt")
    server.load_model(HOME/"models/zebra.pt")
    server.load_model(HOME/"models/best.pt")

    try:
        server.run()
    except KeyboardInterrupt:
        print("Serwer zatrzymany.")
    finally:
        server.cleanup()
