import socket
import threading
import json
import time
import csv

BUFFER_SIZE = 1024
BUFFERED_ITEMS_LIMIT = 10

class ClientThread(threading.Thread):
    def __init__(self, client_address, client_socket):
        threading.Thread.__init__(self)
        self.client_socket = client_socket
        self.buffer = []
        print(f"New connection added: {client_address}")

    def run(self):
        data_buffer = ""  
        json_start = False
        while True:
            data = self.client_socket.recv(BUFFER_SIZE).decode()
            if not data:
                break

            data_buffer += data

            if not json_start and '\r\n\r\n' in data_buffer:
                data_buffer = data_buffer.split('\r\n\r\n', 1)[1]
                json_start = True

            if json_start and '}' in data_buffer:
                json_payload, data_buffer = data_buffer.rsplit('}', 1)
                json_payload += '}'  
                try:
                    parsed_data = json.loads(json_payload)
                    self.buffer.append(parsed_data)
                    json_start = False 
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON from {json_payload} received from {self.client_socket.getpeername()}")

                if len(self.buffer) >= BUFFERED_ITEMS_LIMIT:
                    self.process_buffer()
        if self.buffer:
            self.process_buffer()
        self.client_socket.close()

    def process_buffer(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"data_{timestamp}.csv"

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = self.buffer[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for item in self.buffer:
                writer.writerow(item)
        
        print(f"Buffered data written to {filename}")
        
        self.buffer.clear()

def start_server():
    host = socket.gethostname()
    port = 8000

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)

    print(f"Server started on {host}:{port}")

    while True:
        client_socket, client_address = server_socket.accept()
        new_client = ClientThread(client_address, client_socket)
        new_client.start()

if __name__ == "__main__":
    start_server()
