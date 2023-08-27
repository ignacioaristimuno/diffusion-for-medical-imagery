"""
The training of LoRA weights was based on the following file from the Diffusers library:
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

This implementation is not strictly the same as the original, but takes the most important sections
with fewer functionalities for the sake of simplicity.
"""

from accelerate import Accelerator
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
import math
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from core.settings import custom_logger, get_config
from core.utils import CONCEPTS_FOLDER


class LoRATrainer:
    """
    Class for training the LoRA weights based on the Diffusers library
    """

    def __init__(self, concept_name: str) -> None:
        self.logger = custom_logger(self.__class__.__name__)
        self.concept_name = concept_name
        self.hyperparameters = get_config(key="LoRA")
        self.logger.info(
            f"""Running {self.__class__.__name__} for the concept {concept_name} with hyperparameters: {self.hyperparameters}"""
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using {device.upper()} as device!")
        self.device = torch.device(device)
        self.load_models()

    def load_models(self):
        # Scheduler and tokenizer
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.hyperparameters["model_id"], subfolder="scheduler"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.hyperparameters["model_id"],
            subfolder="tokenizer",
        )

        # Models
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.hyperparameters["model_id"],
            subfolder="text_encoder",
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.hyperparameters["model_id"], subfolder="vae"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.hyperparameters["model_id"], subfolder="unet"
        )

        # Freeze model parameters
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def train(self):
        """Method for initiate the training of LoRA weights"""

        # Accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=self.hyperparameters[
                "gradient_accumulation_steps"
            ],
            mixed_precision=self.hyperparameters["mixed_precision"],
        )
        weight_dtype = self._set_weights_precision(accelerator)

        # Set the correct LoRA layers
        self._set_lora_layers()
        lora_layers = AttnProcsLayers(self.unet.attn_processors)

        # Optimizer
        optimizer = torch.optim.AdamW(
            lora_layers.parameters(),
            lr=self.hyperparameters["learning_rate"],
            betas=(
                self.hyperparameters["adam_beta_1"],
                self.hyperparameters["adam_beta_2"],
            ),
            weight_decay=self.hyperparameters["adam_weight_decay"],
            eps=self.hyperparameters["adam_epsilon"],
        )

        # DataLoader
        train_dataloader = self._load_local_dataset(
            f"{CONCEPTS_FOLDER}/{self.concept_name}", accelerator
        )

        # Scheduler and math around the number of training steps
        override_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.hyperparameters["gradient_accumulation_steps"]
        )
        if self.hyperparameters["max_train_steps"] is None:
            self.hyperparameters["max_train_steps"] = (
                self.hyperparameters["num_train_epochs"] * num_update_steps_per_epoch
            )
            override_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.hyperparameters["lr_scheduler"],
            optimizer=optimizer,
            num_warmup_steps=self.hyperparameters["lr_warmup_steps"]
            * accelerator.num_processes,
            num_training_steps=self.hyperparameters["max_train_steps"]
            * accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`.
        lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            lora_layers, optimizer, train_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.hyperparameters["gradient_accumulation_steps"]
        )
        if override_max_train_steps:
            self.hyperparameters["max_train_steps"] = (
                self.hyperparameters["num_train_epochs"] * num_update_steps_per_epoch
            )

        # Afterwards we recalculate our number of training epochs
        self.hyperparameters["num_train_epochs"] = math.ceil(
            self.hyperparameters["max_train_steps"] / num_update_steps_per_epoch
        )

        # Batch size
        total_batch_size = (
            self.hyperparameters["batch_size"]
            * accelerator.num_processes
            * self.hyperparameters["gradient_accumulation_steps"]
        )

        # Log start of training
        self._log_training_start(train_dataloader, total_batch_size)
        global_step = 0
        first_epoch = 0
        progress_bar = tqdm(
            range(global_step, self.hyperparameters["max_train_steps"]),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        # Training
        for epoch in range(first_epoch, self.hyperparameters["num_train_epochs"]):
            self.logger.info(
                f"Epoch {epoch + 1} of {self.hyperparameters['num_train_epochs']}"
            )
            self.unet.train()
            train_loss = 0.0
            for batch in train_dataloader:
                global_step, step_loss = self._train_one_step(
                    batch,
                    optimizer,
                    lr_scheduler,
                    lora_layers,
                    accelerator,
                    weight_dtype,
                    global_step,
                    progress_bar,
                )
                train_loss += step_loss
                if global_step >= self.hyperparameters["max_train_steps"]:
                    break

            # Validation steps
            self._run_validation_steps(accelerator, epoch, weight_dtype)

        # Save LoRA layers
        self._save_lora_layers(accelerator)

    def _set_lora_layers(self):
        """Method for setting the LoRA layers within the U-Net"""

        lora_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            )

        self.unet.set_attn_processor(lora_attn_procs)
        if torch.cuda.is_available():
            self.unet.enable_xformers_memory_efficient_attention()

    def _set_weights_precision(self, accelerator: Accelerator):
        """Method for setting the weights precision"""

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move U-Net, VAE and Text Encoder to device and cast to weight_dtype
        self.unet.to(accelerator.device, dtype=weight_dtype)
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        return weight_dtype

    def _load_local_dataset(
        self, data_dir: str, accelerator: Accelerator
    ) -> DataLoader:
        """Function for loading the concept's dataset from the data directory"""

        self.logger.info(f"Loading dataset from {data_dir}")
        data_files = {"train": os.path.join(data_dir, "**")}
        train_dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
        )

        # Preprocess dataset
        self.logger.info("Preprocessing dataset...")
        with accelerator.main_process_first():
            train_dataset = train_dataset["train"].with_transform(
                self._preprocess_dataset
            )

        # DataLoaders creation:
        self.logger.info(
            f"Loading dataset in DataLoader with batch size of {self.hyperparameters['batch_size']}"
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.hyperparameters["batch_size"],
        )
        return train_dataloader

    def _tokenize_captions(
        self, examples, caption_column: str = "text", is_train: bool = True
    ):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    def _preprocess_dataset(self, examples, image_column="image"):
        """Function for preprocessing the dataset"""

        # Images
        images = [image.convert("RGB") for image in examples[image_column]]

        # Transformations definitions
        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self.hyperparameters["resolution"],
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(self.hyperparameters["resolution"])
                if self.hyperparameters["center_crop"]
                else transforms.RandomCrop(self.hyperparameters["resolution"]),
                transforms.RandomHorizontalFlip()
                if self.hyperparameters["random_flip"]
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # Preprocess images
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = self._tokenize_captions(examples)
        return examples

    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    def _log_training_start(self, train_dataset, total_batch_size):
        self.logger.info("***** Running training *****")
        self.logger.info(f"Num examples = {len(train_dataset)}")
        self.logger.info(f"Num Epochs = {self.hyperparameters['num_train_epochs']}")
        self.logger.info(
            f"Instantaneous batch size per device = {self.hyperparameters['batch_size']}"
        )
        self.logger.info(
            f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        self.logger.info(
            f"Gradient Accumulation steps = {self.hyperparameters['gradient_accumulation_steps']}"
        )
        self.logger.info(
            f"Total optimization steps = {self.hyperparameters['max_train_steps']}"
        )

    def _train_one_step(
        self,
        batch,
        optimizer,
        lr_scheduler,
        lora_layers,
        accelerator,
        weight_dtype,
        global_step,
        progress_bar,
    ) -> int:
        with accelerator.accumulate(self.unet):
            # Convert images to latent space
            latents = self.vae.encode(
                batch["pixel_values"].to(dtype=weight_dtype)
            ).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            if self.hyperparameters["noise_offset"]:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += self.hyperparameters["noise_offset"] * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1),
                    device=latents.device,
                )

            # Sample a random timestep for each image in the batch
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep (Forward diffusion)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

            # Get the target for loss depending on the prediction type
            if self.hyperparameters["prediction_type"] is not None:
                self.noise_scheduler.register_to_config(
                    prediction_type=self.hyperparameters["prediction_type"]
                )

            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            # Predict the noise residual
            model_pred = self.unet(
                noisy_latents, timesteps, encoder_hidden_states
            ).sample

            # Compute loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(
                loss.repeat(self.hyperparameters["batch_size"])
            ).mean()
            train_loss = (
                avg_loss.item() / self.hyperparameters["gradient_accumulation_steps"]
            )

            # Backpropagation
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = lora_layers.parameters()
                accelerator.clip_grad_norm_(
                    params_to_clip, self.hyperparameters["max_grad_norm"]
                )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            self.logger.info(f"Loss at epoch {global_step}: {train_loss}")
            train_loss = 0.0

            if (
                global_step % self.hyperparameters["checkpointing_steps"] == 0
                and accelerator.is_main_process
            ):
                save_path = os.path.join(
                    self.hyperparameters["output_dir"],
                    f"checkpoint-{global_step}",
                )
                accelerator.save_state(save_path)
                self.logger.info(f"Saved state to {save_path}")

        logs = {
            "step_loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
        }
        progress_bar.set_postfix(**logs)
        return global_step, train_loss

    def _run_validation_steps(self, accelerator, epoch, weight_dtype):
        if (
            not accelerator.is_main_process
            or self.hyperparameters["validation_prompt"] is None
            or epoch % self.hyperparameters["validation_epochs"] != 0
        ):
            return
        self.logger.info(
            f"""Running validation...
            Generating {self.hyperparameters['num_validation_images']} images with prompt:
            {self.hyperparameters['validation_prompt']}.
            """
        )

        # Create pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            self.hyperparameters["model_id"],
            unet=accelerator.unwrap_model(self.unet),
            torch_dtype=weight_dtype,
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # Run inference
        if self.hyperparameters["seed"] is not None:
            generator = generator.manual_seed(self.hyperparameters["seed"])

        generator = torch.Generator(device=accelerator.device)
        images = [
            pipeline(
                self.hyperparameters["validation_prompt"],
                num_inference_steps=30,
                generator=generator,
            ).images[0]
            for _ in range(self.hyperparameters["num_validation_images"])
        ]
        del pipeline
        torch.cuda.empty_cache()

    def _save_lora_layers(self, accelerator):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.unet = self.unet.to(torch.float32)
            self.unet.save_attn_procs(self.hyperparameters["output_dir"])


if __name__ == "__main__":
    concept_name = "cat_toy"

    trainer = LoRATrainer(concept_name)
    trainer.train()
