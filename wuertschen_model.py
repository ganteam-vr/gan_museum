import os
import time
import torch
import torchvision
from PIL import Image
from wuerstchen.vqgan import VQModel
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, T5EncoderModel, CLIPTextModel
from torch.utils.data import DataLoader
from wuerstchen.modules import Paella, EfficientNetEncoder, Prior, DiffNeXt, sample
from wuerstchen.diffuzz import Diffuzz
import transformers
import torchvision.transforms as transforms
import uuid
from pathlib import Path
import numpy as np



class WuertschenModel:

    def embed_clip(self, clip_tokenizer, clip_model, caption, negative_caption="", batch_size=4, device="cuda"):
        clip_tokens = clip_tokenizer([caption] * batch_size, truncation=True, padding="max_length", max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(device)
        clip_text_embeddings = clip_model(**clip_tokens).last_hidden_state

        clip_tokens_uncond = clip_tokenizer([negative_caption] * batch_size, truncation=True, padding="max_length", max_length=clip_tokenizer.model_max_length, return_tensors="pt").to(device)
        clip_text_embeddings_uncond = clip_model(**clip_tokens_uncond).last_hidden_state
        return clip_text_embeddings, clip_text_embeddings_uncond
    

    def decode(self, img_seq):
        return self.vqmodel.decode(img_seq)


    def __init__(self, images_path: str) -> None:
        self.__images_path = images_path

        transformers.utils.logging.set_verbosity_error()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        os.chdir("wuerstchen")

        checkpoint_stage_a = "models/vqgan_f4_v1_500k.pt"
        checkpoint_stage_b = "models/model_v2_stage_b.pt"
        checkpoint_stage_c = "models/model_v2_stage_c_finetune_interpolation.pt"

        effnet_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(768, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
            torchvision.transforms.CenterCrop(768),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        ])


        self.vqmodel = VQModel().to(self.device)
        self.vqmodel.load_state_dict(torch.load(checkpoint_stage_a, map_location=self.device)["state_dict"])
        self.vqmodel.eval().requires_grad_(False)

        self.clip_model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(self.device).eval().requires_grad_(False)
        self.clip_tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

        self.clip_model_b = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").eval().requires_grad_(False).to(self.device)
        self.clip_tokenizer_b = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

        self.diffuzz = Diffuzz(device=self.device)

        pretrained_checkpoint = torch.load(checkpoint_stage_b, map_location=self.device)

        self.effnet = EfficientNetEncoder().to(self.device)
        self.effnet.load_state_dict(pretrained_checkpoint['effnet_state_dict'])
        self.effnet.eval().requires_grad_(False)

        # - LDM Model as generator -
        self.generator = DiffNeXt()
        self.generator.load_state_dict(pretrained_checkpoint['state_dict'])
        self.generator.eval().requires_grad_(False).to(self.device)

        del pretrained_checkpoint

        self.checkpoint = torch.load(checkpoint_stage_c, map_location=self.device)
        self.model = Prior(c_in=16, c=1536, c_cond=1280, c_r=64, depth=32, nhead=24).to(self.device)
        self.model.load_state_dict(self.checkpoint['ema_state_dict'])
        self.model.eval().requires_grad_(False)
        del self.checkpoint

        torch.cuda.empty_cache()
        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
        self.generator = torch.compile(self.generator, mode="reduce-overhead", fullgraph=True)


    def sampling(self, prompt: str) -> Path:

        batch_size = 1
        
        effnet_features_shape = (batch_size, 16, 12, 12)
        effnet_embeddings_uncond = torch.zeros(effnet_features_shape).to(self.device)
        
        caption = prompt
        negative_caption = ""
        prior_inference_steps = {2/3: 20, 0.0: 10}
        prior_cfg = 4
        prior_sampler = "ddpm"

        generator_steps = 12
        generator_cfg = None
        generator_sampler = "ddpm"

        height = 1024
        width = 1024

        clip_text_embeddings, clip_text_embeddings_uncond = self.embed_clip(self.clip_tokenizer, self.clip_model, caption, negative_caption, batch_size, self.device)

        latent_height = 128 * (height // 128) // (1024 // 24)
        latent_width = 128 * (width // 128) // (1024 // 24)
        prior_features_shape = (batch_size, 16, latent_height, latent_width)
        effnet_embeddings_uncond = torch.zeros(effnet_features_shape).to(self.device)
        generator_latent_shape = (batch_size, 4, int(latent_height * (256 / 24)), int(latent_width * (256 / 24)))
        # torch.manual_seed(42)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
            s = time.time()
            t_start = 1.0
            sampled = None
            for t_end, steps in prior_inference_steps.items():
                sampled = self.diffuzz.sample(self.model, {'c': clip_text_embeddings}, x_init=sampled, unconditional_inputs={"c": clip_text_embeddings_uncond}, shape=prior_features_shape,
                                    timesteps=steps, cfg=prior_cfg, sampler=prior_sampler,
                                    t_start=t_start, t_end=t_end)[-1]
                t_start = t_end
            sampled = sampled.mul(42).sub(1)

            print(f"Prior Sampling: {time.time() - s}")

            clip_text_embeddings, clip_text_embeddings_uncond = self.embed_clip(self.clip_tokenizer_b, self.clip_model_b, caption, negative_caption, batch_size, self.device)

            s = time.time()
            sampled_images_original = self.diffuzz.sample(self.generator, {'effnet': sampled, 'clip': clip_text_embeddings},
                                    generator_latent_shape, t_start=1.0, t_end=0.00,
                                    timesteps=generator_steps, cfg=generator_cfg, sampler=generator_sampler,
                                    unconditional_inputs = {
                                    'effnet': effnet_embeddings_uncond, 'clip': clip_text_embeddings_uncond,
                                })[-1]
            print(f"Generator Sampling: {time.time() - s}")

        s = time.time()
        sampled = self.decode(sampled_images_original)
        print(f"Decoder Generation: {time.time() - s}")
        print(f"Prior => CFG: {prior_cfg}, Steps: {sum(prior_inference_steps.values())}, Sampler: {prior_sampler}")
        print(f"Generator => CFG: {generator_cfg}, Steps: {generator_steps}, Sampler: {generator_sampler}")
        print(f"Images Shape: {sampled.shape}")

        # Generate a UUID
        generated_uuid = uuid.uuid4()
        image_numpy = sampled[0].permute(1, 2, 0).cpu()
        image_numpy = np.clip(image_numpy.numpy(), 0, 1)

        # Save PIL image to file
        path = Path(self.__images_path) / Path(f"/generated_image_{str(generated_uuid)[:20]}.png")
        plt.imsave(path, image_numpy)
        print(f"Saved image to {path.as_posix()}")
        return path
        



