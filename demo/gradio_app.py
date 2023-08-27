import gradio as gr

from demo import create_sd_demo


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # <center>Diffusion Models Demo</center>
        <center>Demo for trying out several techniques for generating medical 
        images through Diffusion Models </center>
    """
    )
    with gr.Tabs():
        with gr.TabItem("Base Stable Diffusion"):
            create_sd_demo()
        with gr.TabItem("Textual Inversion"):
            create_sd_demo()  # Placeholder, change it later
