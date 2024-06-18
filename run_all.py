import argparse
from pathlib import Path
import subprocess
import socket
from server_config import get_server_info
from datetime import datetime
import os
import socket
import threading
import time
import json


RUNNING_MODE = False


def execute_shell_command_live(command):
    try:
        # Start the shell command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Read and print the output live
        for line in process.stdout:
            print("STDOUT:", line.strip())

        # Read and print the error output live
        for line in process.stderr:
            print("STDERR:", line.strip())

        # Wait for the process to complete
        process.wait()

    except Exception as e:
        print("Error:", e)


def execute_shell_command(command: str):
    try:
        # Execute the shell command
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Print the output
        print("Output:", result.stdout)

    except subprocess.CalledProcessError as e:
        # Print error if the command failed
        print("Error:", e)
        print("Error output:", e.stderr)



def run_zero123(image_path: str):

    config = 'configs/stable-zero123.yaml' # Relative path!
    data_img_path = image_path
    command = f"python launch.py --config {config} --train --gpu 1 data.image_path={data_img_path} system.prompt_processor.prompt=''"
    execute_shell_command_live(command)
    


def export_zero123():
    model_output_dir = Path("./outputs/zero123-sai/model_runs")
    CONFIG_FILE = f"{model_output_dir}/configs/parsed.yaml"
    CHECKPOINT_FILE=f"{model_output_dir}/ckpts/last.ckpt"
    command = f'python launch.py --config "{CONFIG_FILE}" --export --gpu 1 resume="{CHECKPOINT_FILE}" system.exporter_type=mesh-exporter system.exporter.context_type=cuda'
    execute_shell_command(command)


def create_image(prompt: str) -> Path:
    """
    Create an image using a diffusion model, remove background and return the file path where
    it is stored.

    :prompt: The user prompt as str
    """
    return Path("/home/csai/Documents/gan/threestudio/images/tensor_image.png")



def send_data_to_server(data) -> str:
    host, port = get_server_info()
    
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect to the server
        client_socket.connect((host, port - 1))
        
        # Send data to the server
        client_socket.send(data.encode())
        
        # Receive the response from the server
        response = client_socket.recv(1024).decode()
        print("Response from server:", response)
        return response
        
    except Exception as e:
        print("Error:", " could not connect to server, please verify wuertschen server is running at the configured ip:port")
    
    finally:
        # Close the connection
        client_socket.close()




def create_3d(input_string: str):
    # Create argument parser
    #parser = argparse.ArgumentParser(description='User prompt')

    # Add argument for the string
    #parser.add_argument('-i','--input_string', type=str, help='Input the user prompt as a string', required=True)

    # Parse the command line arguments
    #args = parser.parse_args()

    # Extract the string from the arguments
    #input_string = args.input_string

    # Print the string
    print("Input string:", input_string)

    # Create the 2d image first
    image_path = send_data_to_server("A complete image of " + input_string + " with white background")

    

    

    if image_path is not None:
    
        image_path = Path(image_path)
        # Remove image background
        no_background_path = Path('./images_no_background')
        execute_shell_command(f'transparent-background --source {image_path.as_posix()} --dest {no_background_path.as_posix()}')

        # Call zero123 to convert image to 3d
        run_zero123(image_path=no_background_path / Path(image_path.name[:image_path.name.index(".png")] + "_rgba.png"))

        # Export the file as a mesh
        export_zero123()
        # Now the model exports are saved to /model_runs
        new_path = f'/outputs/zero123-sai/model_run_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        os.rename(os.getcwd() + '/outputs/zero123-sai/model_runs', os.getcwd() + new_path )
        complete_path = 'D:/gan/threestudio' + new_path + "/save/it400-export"
        return complete_path
    else:
        print("Got no 2d image")
    

    
def handle_client(client_socket, address):
    global RUNNING_MODE
    while(RUNNING_MODE):
        pass
    
    RUNNING_MODE = True
    print(f"Connection from {address}")
    
    # Receive data (assuming it's a list of states)
    data = client_socket.recv(1024).decode()
    
    # Split the data by lines
    lines = data.split('\r\n')
    
    # Find the first empty line indicating the end of headers
    for i, line in enumerate(lines):
        if line == '':
            break
    
    # Concatenate the remaining lines (excluding headers)
    tag_name = ''.join(lines[i+1:])
    
    data = tag_name.strip()
    data = json.loads(data)
    data = data.get("tag", "")
        
    print(f"Zero123 model received the following prompt: {data}")
    
    # Process function using states
    start_time = time.time()
    # Call the function
    result = create_3d(input_string=data)
    end_time = time.time()
    # Calculate the execution time
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    
    
    response_body = str(result)
    response = "HTTP/1.1 200 OK\r\n"
    response += "Content-Type: text/plain\r\n"
    response += "Content-Length: {}\r\n".format(len(response_body))
    response += "\r\n"  # End of headers
    response += response_body
    
    
    # Send result back to client
    client_socket.send(response.encode())
    
    # Close connection
    client_socket.close()
    RUNNING_MODE = False
    print(f"Connection with {address} closed")



def start_server():
    global model 
    global RUNNING_MODE
    RUNNING_MODE = False
    
    # Choose any available port
    host, port = get_server_info()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")
    
    try:
        while True:
            client_socket, address = server_socket.accept()
            client_handler = threading.Thread(target=handle_client, args=(client_socket, address))
            client_handler.start()

    except Exception:
        print("Closing server.")
        server_socket.close()


if __name__ == "__main__":
    start_server()


