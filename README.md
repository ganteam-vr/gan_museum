# gan_museum

# Generative AI for 3D Object Creation

Welcome to our project on generative AI for creating 3D objects! Our innovative approach involves developing two interconnected virtual environments that work together to transform text inputs into fully reconstructed 3D models.

## Overview

Our process involves two main phases:

1. **Würstchen Diffusion Model**
2. **Zero123**

### Würstchen Diffusion Model

This phase serves as the starting point for our generative AI workflow. The Würstchen Diffusion Model is a text-to-image model that translates textual descriptions into vivid visual representations. This model lays the groundwork for creating an immersive VR experience by generating detailed images from text inputs.

#### Checkpoints

To use the Würstchen Diffusion Model, you need to download the following checkpoints and place them in the appropriate directory:

```sh
wget https://huggingface.co/dome272/wuerstchen/resolve/main/model_stage_b.pt
wget https://huggingface.co/dome272/wuerstchen/resolve/main/model_stage_c_ema.pt
wget https://huggingface.co/dome272/wuerstchen/resolve/main/vqgan_f4_v1_500k.pt

mv vqgan_f4_v1_500k.pt wuerstchen/models
mv model_* wuerstchen/models

Zero123

After generating the initial images with the Würstchen Diffusion Model, the process moves to the Zero123 phase. Zero123 is a model that transforms these 2D images into full 3D reconstructions from a single RGB image using advanced camera viewpoint adjustments. This phase enhances the immersive experience by creating detailed and accurate 3D models.
Checkpoint

To use the Zero123 model, download the checkpoint from the following link and save it to the specified directory:
Download the checkpoint from [https://huggingface.co/stabilityai/stable-zero123/blob/main/stable_zero123.ckpt](https://huggingface.co/stabilityai/stable-zero123/blob/main/stable_zero123.ckpt) and save it to `threestudio/load/zero123`
