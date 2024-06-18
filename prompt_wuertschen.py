
import argparse
import socket
from server_config import get_server_info
import cv2
import numpy as np


def send_data_to_server(data) -> str:
    host, port = get_server_info()
    
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect to the server
        client_socket.connect((host, port))
        
        # Send data to the server
        client_socket.send(data.encode())
        
        # Receive the response from the server
        response = client_socket.recv(1024).decode()
        print("Response from server:", response)
        return response
        
    except Exception as e:
        print("Error:", e)
    
    finally:
        # Close the connection
        client_socket.close()


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='User prompt')

    # Add argument for the string
    parser.add_argument('-i','--input_string', type=str, help='Input the user prompt as a string', required=True)

    # Parse the command line arguments
    args = parser.parse_args()

    # Extract the string from the arguments
    input_string = args.input_string

    # Print the string
    print("Input string:", input_string)

    

    # Create the 2d image first
    image_path = send_data_to_server(input_string + " with black background")
    print(f"placed image at {image_path}")
    return image_path


def remove_background(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Create a mask for the foreground object
    mask = np.zeros(image.shape[:2], np.uint8)

    # Specify the rectangle enclosing the foreground object
    rect = (50, 50, 450, 290)

    # Run GrabCut algorithm to segment the foreground object
    cv2.grabCut(image, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)

    # Create a mask where all background and probable background pixels are marked as 0
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the original image
    result = image * mask2[:, :, np.newaxis]

    return result

# Example usage:
image_path = main()

