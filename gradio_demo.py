import gradio as gr
from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline
import torch

def generate_images(prompt, negative_prompt, height, width, num_inference_steps, guidance_scale, cosine_scale_1, cosine_scale_2, cosine_scale_3, sigma, view_batch_size, stride, seed):
    model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DemoFusionSDXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(int(seed))

    images = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
                  height=int(height), width=int(width), view_batch_size=int(view_batch_size), stride=int(stride),
                  num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                  cosine_scale_1=cosine_scale_1, cosine_scale_2=cosine_scale_2, cosine_scale_3=cosine_scale_3, sigma=sigma,
                  multi_decoder=True, show_image=False
                 )

    return [image for _, image in enumerate(images)]

iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Negative Prompt", value="blurry, ugly, duplicate, poorly drawn, deformed, mosaic"),
        gr.Slider(minimum=1024, maximum=4096, step=1024, value=2048, label="Height"),
        gr.Slider(minimum=1024, maximum=4096, step=1024, value=2048, label="Width"),
        gr.Slider(minimum=10, maximum=100, step=1, value=50, label="Num Inference Steps"),
        gr.Slider(minimum=1, maximum=20, step=0.1, value=7.5, label="Guidance Scale"),
        gr.Slider(minimum=0, maximum=5, step=0.1, value=3, label="Cosine Scale 1"),
        gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 2"),
        gr.Slider(minimum=0, maximum=5, step=0.1, value=1, label="Cosine Scale 3"),
        gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.8, label="Sigma"),
        gr.Slider(minimum=4, maximum=32, step=4, value=16, label="View Batch Size"),
        gr.Slider(minimum=8, maximum=96, step=8, value=64, label="Stride"),
        gr.Number(label="Seed", value=2013)
    ],
    outputs=gr.Gallery(label="Generated Images"),
    title="DemoFusion Gradio Demo",
    description="Generate images with the DemoFusion SDXL Pipeline."
)

iface.launch(server_name='114.113.235.51', server_port=8890)
