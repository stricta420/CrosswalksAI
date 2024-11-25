import cv2
import socket
import pickle
import numpy as np
import os
from pathlib import Path


# Konfiguracja klienta
SERVER_IP = '127.0.0.1'  # Publiczny adres IP serwera
PORT = 5000
HEADERSIZE = 10
HOME = Path(__file__).resolve().parent.parent

def main():
    # Połączenie z serwerem
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_IP, PORT))
        print(f"Połączono z serwerem {SERVER_IP}:{PORT}")
    except Exception as e:
        print(f"Błąd podczas łączenia z serwerem: {e}")
        return

    # Otwieranie kamery
    cap = cv2.VideoCapture(HOME/"video/wideo2.mp4")
    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        return

    try:
        while True:
            # Pobieranie klatki z kamery
            ret, frame = cap.read()
            if not ret:
                print("Nie można odczytać klatki z kamery")
                break

            # Serializacja obrazu
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = pickle.dumps(buffer)
            frame_data = bytes(f"{len(frame_data):<{HEADERSIZE}}", 'utf-8') + frame_data

            # Wysyłanie obrazu do serwera
            client_socket.send(frame_data)

            # Odbiór analizy obrazu
            full_msg = b''
            new_msg = True

            while True:
                data = client_socket.recv(4096)
                if new_msg:
                    if len(data) < HEADERSIZE:
                        print("Niepełne dane nagłówka, przerywam.")
                        break
                    msglen = int(data[:HEADERSIZE])
                    new_msg = False

                full_msg += data

                if len(full_msg) - HEADERSIZE == msglen:
                    print("Odebrano pełny komunikat.")
                    analyzed_frame_data = pickle.loads(full_msg[HEADERSIZE:])
                    full_msg = b''
                    new_msg = True
                    break
            
            # Dekodowanie analizowanego obrazu
            analyzed_frame = cv2.imdecode(np.frombuffer(analyzed_frame_data, np.uint8), cv2.IMREAD_COLOR)

            # Wyświetlanie wyników analizy
            cv2.imshow('Oryginalny Obraz', frame)
            cv2.imshow('Analiza', analyzed_frame)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        print(f"Błąd w trakcie działania: {e}")
    finally:
        # Zamknięcie połączenia i kamer
        cap.release()
        client_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
