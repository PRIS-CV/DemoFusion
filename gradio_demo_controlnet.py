import gradio as gr
from diffusers import ControlNetModel, AutoencoderKL
from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline
from pipeline_demofusion_sdxl_controlnet import DemoFusionSDXLControlNetPipeline
from gradio_imageslider import ImageSlider
import torch, gc
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

def load_and_process_image(pil_image):
    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = transform(pil_image)
    image = image.unsqueeze(0).half()
    return image


def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        pad_w = 0
        pad_h = (w - h) // 2
        new_image.paste(image, (0, pad_h))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        pad_w = (h - w) // 2
        pad_h = 0
        new_image.paste(image, (pad_w, 0))
        return new_image

def generate_images(prompt, negative_prompt, controlnet_conditioning_scale, height, width, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, sigma, view_batch_size, stride, seed, input_image):
    padded_image = pad_image(input_image).resize((1024, 1024)).convert("RGB")
    image_lr = load_and_process_image(padded_image).to('cuda')
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained("madebyollin/stable-diffusion-xl-base-1.0/vae-fix", torch_dtype=torch.float16)
    pipe = DemoFusionSDXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(int(seed))
    # get canny image
    canny_image = np.array(padded_image)
    canny_image = cv2.Canny(canny_image, 100, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)
    images = pipe(prompt, negative_prompt=negative_prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, 
                  condition_image=canny_image, generator=generator,
                  height=int(height), width=int(width), view_batch_size=int(view_batch_size), stride=int(stride),
                  num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                  cosine_scale_1=cosine_scale_1, cosine_scale_2=cosine_scale_2, cosine_scale_3=cosine_scale_3, sigma=sigma,
                  multi_decoder=True, show_image=False, lowvram=False
                 )
    for i, image in enumerate(images):
      image.save('image_'+str(i)+'.png')
    pipe = None
    gc.collect()
    torch.cuda.empty_cache()
    return (canny_image, images[-1])

with gr.Blocks(title=f"DemoFusion") as demo:
    with gr.Column():
      with gr.Row():
        with gr.Group():
          image_input = gr.Image(type="pil", label="Input Image")
          prompt = gr.Textbox(label="Prompt", value="")
          negative_prompt = gr.Textbox(label="Negative Prompt", value="blurry, ugly, duplicate, poorly drawn, deformed, mosaic")
          controlnet_conditioning_scale = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label="ControlNet Conditioning Scale")
          width = gr.Slider(minimum=1024, maximum=4096, step=1024, value=2048, label="Width")
          height = gr.Slider(minimum=1024, maximum=4096, step=1024, value=2048, label="Height")
          num_inference_steps = gr.Slider(minimum=10, maximum=100, step=1, value=50, label="Num Inference Steps")
          guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.1, value=7.5, label="Guidance Scale")
          cosine_scale_1 = gr.Slider(minimum=0, maximum=5, step=0.1, value=3, label="Cosine Scale 1")
          cosine_scale_2 = gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 2")
          cosine_scale_3 = gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 3")
          sigma = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.8, label="Sigma")
          view_batch_size = gr.Slider(minimum=4, maximum=32, step=4, value=16, label="View Batch Size")
          stride = gr.Slider(minimum=8, maximum=96, step=8, value=64, label="Stride")
          seed = gr.Number(label="Seed", value=2013)
          button = gr.Button()
        output_images = ImageSlider(show_label=False)
    button.click(fn=generate_images, inputs=[prompt, negative_prompt, controlnet_conditioning_scale, height, width, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, sigma, view_batch_size, stride, seed, image_input], outputs=[output_images], show_progress=True)
demo.queue().launch(inline=False, share=True, debug=True)
