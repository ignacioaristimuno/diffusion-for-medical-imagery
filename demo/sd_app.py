import gradio as gr

from core.models import StableDiffusionModel
from core.settings import get_config


configs = get_config("StableDiffusionModel")
model = StableDiffusionModel(**(configs))


def create_sd_demo():
    """Function for creating a demo for the base Stable Diffusion model"""

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("## Base Stable Diffusion 1.4 model")
        with gr.Row():
            with gr.Column():
                # Input parameters
                prompt = gr.Textbox(label="Prompt")
                neg_prompt = gr.Textbox(
                    label="Negative prompt",
                    value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                )
                n_images = gr.Slider(label="Number of images", minimum=1, maximum=4)
                height = gr.Slider(
                    label="Image height", minimum=256, maximum=512, value=512
                )
                width = gr.Slider(
                    label="Image width", minimum=256, maximum=512, value=512
                )
                run_button = gr.Button(label="Generate")

                # Advanced options
                with gr.Accordion("Advanced options", open=False):
                    num_steps = gr.Slider(
                        label="Steps", minimum=1, maximum=100, value=25, step=1
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=0.1,
                        maximum=30.0,
                        value=9.0,
                        step=0.1,
                    )

            # Output
            with gr.Column():
                result = gr.Gallery(
                    label="Generated images", show_label=False, elem_id="gallery"
                ).style(grid=2, height="auto")

        # Logic
        inputs = [
            prompt,
            neg_prompt,
            n_images,
            height,
            width,
            num_steps,
            guidance_scale,
        ]
        run_button.click(fn=model.generate_images, inputs=inputs, outputs=result)
    return demo
