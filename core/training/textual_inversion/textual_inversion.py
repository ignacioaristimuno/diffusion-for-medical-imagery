import accelerate
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
import itertools
import math
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from core.base_objects import ConceptType
from core.settings import custom_logger, get_config
from core.dataset.datasets import TextualInversionDataset
from core.utils import (
    load_json,
    show_image_grid,
    store_images_from_urls,
    CONCEPTS_FOLDER,
)


class TextualInversionTrainer:
    """
    Class for handling the finetuning the Stable Diffusion model using
    Textual Inversion for teaching the model a specific concept.

    https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb
    """

    def __init__(
        self, concept_name: str, placeholder_token: str, initializer_token: str
    ) -> None:
        self.logger = custom_logger(self.__class__.__name__)
        self.hyperparameters = get_config(key="TextualInversion")
        self.model_id = self.hyperparameters["model_id"]
        self.concept_name = concept_name
        self.initializer_token = initializer_token
        self.placeholder_token = placeholder_token

        os.makedirs(self.hyperparameters["output_dir"], exist_ok=True)

        self.images_folder = f"{CONCEPTS_FOLDER}/{concept_name}"
        self.prompt_templates = load_json(
            "core/training/textual_inversion/prompt_templates.json"
        )
        self._initialize_tokenizer()
        self._get_special_tokens(initializer_token)
        self._load_diffusion_model()

    def _initialize_tokenizer(self) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id,
            subfolder="tokenizer",
        )
        # Add the placeholder token in tokenizer
        num_added_tokens = self.tokenizer.add_tokens(self.placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {self.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

    def _get_special_tokens(self, initializer_token: str) -> None:
        token_ids = self.tokenizer.encode(initializer_token, add_special_tokens=False)
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        self.initializer_token_id = token_ids[0]
        self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
            self.placeholder_token
        )

    def _load_diffusion_model(self) -> None:
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        )
        self.noise_scheduler = DDPMScheduler.from_config(
            self.model_id, subfolder="scheduler"
        )

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        token_embeds[self.placeholder_token_id] = token_embeds[
            self.initializer_token_id
        ]

    @staticmethod
    def _freeze_params(params) -> None:
        for param in params:
            param.requires_grad = False

    def _freeze_models(self) -> None:
        # Freeze VAE and U-Net
        self.freeze_params(self.vae.parameters())
        self.freeze_params(self.unet.parameters())

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            self.text_encoder.text_model.encoder.parameters(),
            self.text_encoder.text_model.final_layer_norm.parameters(),
            self.text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        self.freeze_params(params_to_freeze)

    def finetune_model(self):
        batch_size = self.hyperparameters["batch_size"]
        train_dataset = self.create_dataset(self.images_folder)
        train_dataloader = self.create_dataloader(
            train_dataset, batch_size, shuffle=True
        )

        # Training preparation
        accelerator = Accelerator(
            gradient_accumulation_steps=self.hyperparameters[
                "gradient_accumulation_steps"
            ],
            mixed_precision=self.hyperparameters["mixed_precision"],
        )

        if self.hyperparameters["gradient_checkpointing"]:
            self.text_encoder.gradient_checkpointing_enable()
            self.unet.enable_gradient_checkpointing()

        if self.hyperparameters["scale_lr"]:
            learning_rate = (
                self.hyperparameters["learning_rate"]
                * self.hyperparameters["gradient_accumulation_steps"]
                * self.hyperparameters["batch_size"]
                * accelerator.num_processes
            )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.text_encoder.get_input_embeddings().parameters(),  # Only optimize the embeddings
            lr=learning_rate,
        )

        text_encoder, optimizer, train_dataloader = accelerator.prepare(
            self.text_encoder, optimizer, train_dataloader
        )

        # Precision
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move VAE and U-Net to device
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.unet.to(accelerator.device, dtype=weight_dtype)

        self.vae.eval()  # We don't train it
        self.unet.train()  # To enable activation checkpointing

        # We need to recalculate our total training steps as the size of the training dataloader may have changed
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.hyperparameters["gradient_accumulation_steps"]
        )
        num_train_epochs = math.ceil(
            self.hyperparameters["max_train_steps"] / num_update_steps_per_epoch
        )
        total_batch_size = (
            batch_size
            * accelerator.num_processes
            * self.hyperparameters["gradient_accumulation_steps"]
        )

        # Training loop
        progress_bar = self._initialize_logs_and_progress_bar(
            train_dataset, accelerator, batch_size, total_batch_size
        )
        global_step = 0
        for epoch in range(num_train_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_train_epochs}")
            text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                global_step = self._train_one_step(
                    accelerator,
                    batch,
                    optimizer,
                    text_encoder,
                    weight_dtype,
                    progress_bar,
                    global_step,
                )
                if global_step >= self.hyperparameters["max_train_steps"]:
                    break
            accelerator.wait_for_everyone()

        # Save the model
        self.save_trained_pipeline(accelerator, text_encoder)

    def create_dataset(self, save_path: str) -> DataLoader:
        return TextualInversionDataset(
            data_root=save_path,
            tokenizer=self.tokenizer,
            size=self.vae.sample_size,
            placeholder_token=self.placeholder_token,
            repeats=100,
            learnable_property=ConceptType.OBJECT,
            center_crop=False,
            set_split="train",
        )

    def create_dataloader(self, dataset, batch_size: int = 1, shuffle: bool = False):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _initialize_logs_and_progress_bar(
        self, train_dataset, accelerator, batch_size, total_batch_size
    ) -> None:
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(train_dataset)}")
        self.logger.info(f"  Instantaneous batch size per device = {batch_size}")
        self.logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        self.logger.info(
            f"  Gradient Accumulation steps = {self.hyperparameters['gradient_accumulation_steps']}"
        )
        self.logger.info(
            f"  Total optimization steps = {self.hyperparameters['max_train_steps']}"
        )
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(self.hyperparameters["max_train_steps"]),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")
        return progress_bar

    def save_progress(self, text_encoder, accelerator, save_path):
        self.logger.info("Saving embeddings")
        learned_embeds = (
            accelerator.unwrap_model(text_encoder)
            .get_input_embeddings()
            .weight[self.placeholder_token_id]
        )
        learned_embeds_dict = {self.placeholder_token: learned_embeds.detach().cpu()}
        torch.save(learned_embeds_dict, save_path)

    def _train_one_step(
        self,
        accelerator,
        batch,
        optimizer,
        text_encoder,
        weight_dtype,
        progress_bar,
        global_step,
    ):
        with accelerator.accumulate(text_encoder):
            # Convert images to latent space
            latents = (
                self.vae.encode(batch["pixel_values"].to(dtype=weight_dtype))
                .latent_dist.sample()
                .detach()
            )
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                self.noise_scheduler.num_train_timesteps,
                (bsz,),
                device=latents.device,
            ).long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # Predict the noise residual
            noise_pred = self.unet(
                noisy_latents, timesteps, encoder_hidden_states.to(weight_dtype)
            ).sample

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            loss = (
                F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
            )
            accelerator.backward(loss)

            # Zero out the gradients for all token embeddings except the newly added
            # embeddings for the concept, as we only want to optimize the concept embeddings
            if accelerator.num_processes > 1:
                grads = text_encoder.module.get_input_embeddings().weight.grad
            else:
                grads = text_encoder.get_input_embeddings().weight.grad
            # Get the index for tokens that we want to zero the grads for
            index_grads_to_zero = (
                torch.arange(len(self.tokenizer)) != self.placeholder_token_id
            )
            grads.data[index_grads_to_zero, :] = grads.data[
                index_grads_to_zero, :
            ].fill_(0)

            optimizer.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            if global_step % self.hyperparameters["save_steps"] == 0:
                save_path = os.path.join(
                    self.hyperparameters["output_dir"],
                    f"learned_embeds-step-{global_step}.bin",
                )
                self.save_progress(text_encoder, accelerator, save_path)

        logs = {"loss": loss.detach().item()}
        progress_bar.set_postfix(**logs)
        return global_step

    def save_trained_pipeline(self, accelerator, text_encoder) -> None:
        # Create the pipeline using using the trained modules and save it
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=self.tokenizer,
                vae=self.vae,
                unet=self.unet,
            )
            pipeline.save_pretrained(self.hyperparameters["output_dir"])

            # Also save the newly trained embeddings
            save_path = os.path.join(
                self.hyperparameters["output_dir"],
                f"{self.concept_name}_embeddings.bin",
            )
            self.save_progress(text_encoder, accelerator, save_path)

    def run(self):
        accelerate.notebook_launcher(self.finetune_model, args=())

        for param in itertools.chain(
            self.unet.parameters(), self.text_encoder.parameters()
        ):
            if param.grad is not None:
                del param.grad  # Free some memory
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Parameters
    concept_name = "cat_toy"
    what_to_teach = ConceptType.OBJECT  # ["object", "style"]
    placeholder_token = "<cat-toy>"
    initializer_token = "toy"  # A word that can summarise what your new concept is, to be used as a starting point

    # Store images
    finetuning_images = [
        "https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg",
        "https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg",
    ]

    store_images_from_urls(finetuning_images, concept_name=concept_name)

    # Check images
    images = []
    concept_path = f"{CONCEPTS_FOLDER}/{concept_name}"
    for file_path in os.listdir(concept_path):
        try:
            image_path = os.path.join(concept_path, file_path)
            images.append(Image.open(image_path).resize((512, 512)))
        except Exception:
            print(
                f"{image_path} is not a valid image, please make sure to remove this file from the directory otherwise the training could fail."
            )

    show_image_grid(images, 1, len(images))

    # Finetune
    trainer = TextualInversionTrainer(
        concept_name, placeholder_token, initializer_token
    )
    trainer.run()
