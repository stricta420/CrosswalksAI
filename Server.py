import socket
import pickle
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLOv10
from pathlib import Path

HOME = Path(__file__).resolve().parent

class VideoServer:
    def __init__(self, host, port, headersize=10):
        self.host = host
        self.port = port
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

    def setup_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind((self.host, self.port))
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
                image = self.bounding_box_annotator.annotate(scene=image, detections=detections)
                image = self.label_annotator.annotate(scene=image, detections=detections)
            return image
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
                        analyzed_frame = self.analyze_image(frame)

                        _, buffer = cv2.imencode('.jpg', analyzed_frame)
                        analyzed_data = pickle.dumps(buffer)
                        analyzed_data = bytes(f"{len(analyzed_data):<{self.headersize}}", 'utf-8') + analyzed_data
                        conn.send(analyzed_data)
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

    server = VideoServer(HOST, PORT)
    server.load_model(HOME/"models/human.pt")
    server.load_model(HOME/"models/zebra.pt")
    server.load_model(HOME/"models/best.pt")

    try:
        server.run()
    except KeyboardInterrupt:
        print("Serwer zatrzymany.")
    finally:
        server.cleanup()
