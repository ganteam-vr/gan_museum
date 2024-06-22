# Generative AI for 3D Object Creation

Welcome to our project on generative AI for creating 3D objects! Our innovative approach involves developing two interconnected virtual environments that work together to transform text inputs into fully reconstructed 3D models.

## Overview

Our process involves two main phases:

1. **Würstchen Diffusion Model**
2. **Zero123**

### Würstchen Diffusion Model

This phase serves as the starting point for our generative AI workflow. The Würstchen Diffusion Model is a text-to-image model that translates textual descriptions into vivid visual representations. This model lays the groundwork for creating an immersive VR experience by generating detailed images from text inputs.

### Zero123

After generating the initial images with the Würstchen Diffusion Model, the process moves to the Zero123 phase. Zero123 is a model that transforms these 2D images into full 3D reconstructions from a single RGB image using advanced camera viewpoint adjustments. This phase enhances the immersive experience by creating detailed and accurate 3D models.
Checkpoint. The Zero123 model is made availabe with the threestudio framework.

## Requirements
...

### Installation instructions

#### 1. Clone the wuerstchen repository
```sh
# It is expected to overwrite the placeholder wuerstchen directory with the following repo
git clone https://github.com/dome272/Wuerstchen.git
```

#### 2. Checkpoints for Wuerstchen

To use the Wuerstchen Diffusion Model, you need to download the following checkpoints and place them in the appropriate directory:

```sh
cd wuerstchen/models
wget https://huggingface.co/dome272/wuerstchen/resolve/main/model_stage_b.pt
wget https://huggingface.co/dome272/wuerstchen/resolve/main/model_stage_c_ema.pt
wget https://huggingface.co/dome272/wuerstchen/resolve/main/vqgan_f4_v1_500k.pt
```


#### 3. Clone our fork of the threestudio repository
```sh
# It is expected to overwrite the placeholder threestudio directory with the following repo
git clone https://github.com/ganteam-vr/threestudio.git
```

#### 4. Checkpoints for Zero123

To use the Zero123 model, download the checkpoint from the following link and save it to the specified directory:
Download the checkpoint from 
```sh
https://huggingface.co/stabilityai/stable-zero123/blob/main/stable_zero123.ckpt
```
and save it to `threestudio/load/zero123`

#### 5. Directory Structure

Ensure your project directory is structured as follows to accommodate the checkpoints:

```lua

project_root/
├── wuerstchen/
│   ├── models/
│   │   ├── model_stage_b.pt
│   │   ├── model_stage_c_ema.pt
│   │   ├── vqgan_f4_v1_500k.pt
├── threestudio/
│   ├── load/
│   │   ├── zero123/
│   │   │   ├── stable_zero123.ckpt
```
### Run instructions

#### Running Instructions for WSL Users

To run the Wuerstchen and Threestudio environments on WSL, follow these steps:

##### Step 1: Start Two Ubuntu Instances as Admin

1. Open Windows Terminal:
   - Launch the Windows Terminal or your preferred terminal emulator.

2. Start the First Ubuntu Instance:
   - Open a new tab or window in the terminal.
   - Run the following command to start the first Ubuntu instance as an administrator:
     ```sh
     wsl -d Ubuntu-20.04
     ```
   - Note: Replace `Ubuntu-20.04` with the name of your WSL distribution if different.

3. Start the Second Ubuntu Instance:
   - Open another new tab or window in the terminal.
   - Run the same command to start the second Ubuntu instance as an administrator:
     ```sh
     wsl -d Ubuntu-20.04
     ```

##### Step 2: Activate the Conda Environments

1. Activate Threestudio Environment in the First Instance:
   - In the first Ubuntu instance, activate the conda environment for Threestudio:
     ```sh
     conda activate threestudio
     ```

2. Activate Wuerstchen Environment in the Second Instance:
   - In the second Ubuntu instance, activate the conda environment for Wuerstchen:
     ```sh
     conda activate wuerstchen
     ```

##### Step 3: Start the Servers

1. Start the Wuerstchen Server:
   - In the second Ubuntu instance (with the Wuerstchen environment activated), run the following command to start the Wuerstchen server:
     ```sh
     # Important: Run this command in the root directory of this repo
     python run_wuerstchen.py
     ```

3. Start the Main Server:
   - In the first Ubuntu instance (with the Threestudio environment activated), run the following command to start the main server:
     ```sh
     # Important: Run this command in the root directory of this repo
     python run_all.py
     ```

By following these instructions, you will have the Wuerstchen server and the main server running in their respective conda environments on WSL.

### Generating objects
To generate objects send a JSON formatted HTTP Request to the configured server ip and port in the server_config.yaml
The JSON input should follow this format:

```json
{
  "tag": "PROMPT"
}

