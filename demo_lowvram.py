
'''
Installation on Windows for GPU with 8 Gb of VRAM and xformers:

git clone "https://github.com/PRIS-CV/DemoFusion"
cd DemoFusion
python -m venv venv
venv\Scripts\activate
pip install -U "xformers==0.0.22.post7+cu118" --index-url https://download.pytorch.org/whl/cu118
pip install "diffusers==0.21.4" "matplotlib==3.8.2" "transformers==4.35.2" "accelerate==0.25.0"
'''

from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline

import torch
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DemoFusionSDXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16, vae=vae)
pipe = pipe.to("cuda")

prompt = "Envision a portrait of an elderly woman, her face a canvas of time, framed by a headscarf with muted tones of rust and cream. Her eyes, blue like faded denim. Her attire, simple yet dignified."
negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"

images = pipe(prompt, negative_prompt=negative_prompt,
              height=2048, width=2048, view_batch_size=4, stride=64,
              num_inference_steps=40, guidance_scale=7.5,
              cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
              multi_decoder=True, show_image=False, lowvram=True
             )

for i, image in enumerate(images):
    image.save('image_'+str(i)+'.png')
