# gan_museum

## Zero123 Checkpoint

Download the checkpoint from https://huggingface.co/stabilityai/stable-zero123/blob/main/stable_zero123.ckpt and save it to threestudio/load/

## Wuertschen checkpoint (instructions from stage_c notebook in official wuerstchen repo)
wget https://huggingface.co/dome272/wuerstchen/resolve/main/model_stage_b.pt
wget https://huggingface.co/dome272/wuerstchen/resolve/main/model_stage_c_ema.pt
wget https://huggingface.co/dome272/wuerstchen/resolve/main/vqgan_f4_v1_500k.pt

mv vqgan_f4_v1_500k.pt wuerstchen/models
mv model_* wuerstchen/models
