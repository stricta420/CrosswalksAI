import cv2
import socket
import pickle

# Konfiguracja klienta
SERVER_IP = '127.0.0.1'  # Publiczny adres IP serwera
PORT = 5000
HEADERSIZE = 10
def main():
    # Połączenie z serwerem
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_IP, PORT))
    print(f"Połączono z serwerem {SERVER_IP}:{PORT}")

    # Otwieranie kamery
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        return

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
        #print(frame_data)
        # Wysyłanie obrazu do serwera
        client_socket.send(frame_data)

        # Odbiór analizy obrazu
        full_msg = b''

        
    new_msg = True
    while True:
        msg = s.recv(16)
        if new_msg:
            print("new msg len:",msg[:HEADERSIZE])
            msglen = int(msg[:HEADERSIZE])
            new_msg = False

        print(f"full message length: {msglen}")

        full_msg += msg

        print(len(full_msg))

        if len(full_msg)-HEADERSIZE == msglen:
            print("full msg recvd")
            print(full_msg[HEADERSIZE:])
            print(pickle.loads(full_msg[HEADERSIZE:]))
            new_msg = True
            full_msg = b""
        data = client_socket.recv(60000)
        analyzed_frame_data = pickle.loads(data)
        analyzed_frame = cv2.imdecode(np.frombuffer(analyzed_frame_data, np.uint8), cv2.IMREAD_GRAYSCALE)

        # Wyświetlanie wyników analizy
        cv2.imshow('Oryginalny Obraz', frame)
        cv2.imshow('Analiza', analyzed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Zamknięcie połączenia i kamer
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()