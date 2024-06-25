import socket
import threading
from wuertschen_model import WuertschenModel
from server_config import get_server_info
import time
import argparse


model: WuertschenModel = None


def process_function(prompt: str):
    # Example function using the states
    # This could be any function that uses the states
    
    return model.sampling(prompt)

def handle_client(client_socket, address):
    print(f"Connection from {address}")
    
    # Receive data (assuming it's a list of states)
    data = client_socket.recv(1024).decode()

    print(f"Wuertschen model received the following prompt: {data}")
    
    # Process function using states
    start_time = time.time()
    # Call the function
    result = process_function(prompt=data)
    end_time = time.time()
    # Calculate the execution time
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    
    # Send result back to client
    client_socket.send(str(result).encode())
    
    # Close connection
    client_socket.close()
    print(f"Connection with {address} closed")

def start_server(images_path: str):
    global model 
    
    # Choose any available port
    host, port = get_server_info()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port - 1))
    server_socket.listen(5)
    model= WuertschenModel(images_path)
    print(f"Server listening on {host}:{port - 1}")
    
    try:
        while True:
            client_socket, address = server_socket.accept()
            client_handler = threading.Thread(target=handle_client, args=(client_socket, address))
            client_handler.start()

    except Exception:
        print("Closing server.")
        server_socket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give the path to directory where images will be stored.")
    parser.add_argument('images_path', type=str, help='The path to the images directory')

    args = parser.parse_args()
    start_server(args.images_path)

    