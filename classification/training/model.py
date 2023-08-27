import numpy as np
import os
from typing import Any, Dict, List, Tuple

import tensorflow as tf

# from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    GlobalAveragePooling2D,
    Input,
    Dropout,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from core.settings import custom_logger, get_config


def get_cnn_settings() -> Dict[str, Any]:
    """Function for retrieving the configs for the MultiinstanceCNN class"""

    configs = get_config(
        "TrainCNN", config_path="classification/settings/base_config.yml"
    )
    settings = {
        "images_shape": configs["ImagesShape"],
        "batch_size": configs["BatchSize"],
        "first_finetuning_epochs": configs["FirstFinetuningEpochs"],
        "second_finetuning_epochs": configs["SecondFinetuningEpochs"],
        "double_finetuning": configs["DoubleFinetuning"],
        "first_finetuning_lr": configs["FirstFinetuningLR"],
        "second_finetuning_lr": configs["SecondFinetuningLR"],
        "checkpoints_first_finetuning_path": configs["CheckpointsFirstFinetuningPath"],
        "checkpoints_second_finetuning_path": configs[
            "CheckpointsSecondFinetuningPath"
        ],
    }
    return settings


class SkinLesionCNN:
    """
    Class for defining the CNN architecture for binary classification.

    The training will consist of a first pretraining freezing the backbone (EfficientNetB0)
    and adding a few dense layers after. Then, a finetuning step is being held by
    unfreezing the backbone for training the whole network.
    """

    def __init__(
        self,
        images_shape: List[int],
        batch_size: int,
        double_finetuning: bool,
        first_finetuning_epochs: int,
        second_finetuning_epochs: int,
        first_finetuning_lr: float,
        second_finetuning_lr: float,
        checkpoints_first_finetuning_path: str,
        checkpoints_second_finetuning_path: str,
    ) -> None:
        self.logger = custom_logger(self.__class__.__name__)

        # Configs
        self.batch_size = batch_size
        self.images_shape = images_shape
        self.double_finetuning = double_finetuning
        self.first_finetuning_epochs = first_finetuning_epochs
        self.second_finetuning_epochs = second_finetuning_epochs
        self.first_finetuning_lr = first_finetuning_lr
        self.second_finetuning_lr = second_finetuning_lr
        self.checkpoints_first_finetuning_path = checkpoints_first_finetuning_path
        self.checkpoints_second_finetuning_path = checkpoints_second_finetuning_path

        # Create directories for model checkpoints
        os.makedirs(
            "/".join(checkpoints_first_finetuning_path.split("/")[:-1]), exist_ok=True
        )
        os.makedirs(
            "/".join(checkpoints_second_finetuning_path.split("/")[:-1]), exist_ok=True
        )

    def train_model(
        self,
        first_train_data_gen: ImageDataGenerator,
        first_val_data_gen: ImageDataGenerator,
        second_train_data_gen: ImageDataGenerator,
        second_val_data_gen: ImageDataGenerator,
    ) -> Tuple[Model, Dict[str, Any]]:
        """Method for training the CNN in two steps: a pretraining and a finetuning"""

        # Create base model
        self.logger.info("Creating model...")
        model = self._get_model()

        # First finetuning (with generated images)
        if self.double_finetuning:
            self._compile_model(model, step="first_finetuning")
            first_finetuning_metrics = self._train_model_step(
                model, first_train_data_gen, first_val_data_gen, step="first_finetuning"
            )

        # Second finetuning
        self._compile_model(model, step="second_finetuning")
        second_finetuning_metrics = self._train_model_step(
            model, second_train_data_gen, second_val_data_gen, step="second_finetuning"
        )

        # Build metrics dictionary
        metrics = {
            "second_finetuning_metrics": second_finetuning_metrics,
        }
        if self.double_finetuning:
            metrics["first_finetuning_metrics"] = first_finetuning_metrics
        return model, metrics

    def _get_model(self) -> Model:
        # Backbone
        backbone = MobileNet(
            weights="imagenet", input_shape=tuple(self.images_shape), include_top=False
        )
        backbone.trainable = True

        # Network definition
        inputs = Input(shape=tuple(self.images_shape))
        x = backbone(inputs)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)

        x = Dense(256, activation="leaky_relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(128, activation="leaky_relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)

        out = Dense(3)(x)
        model = Model(inputs=inputs, outputs=out)
        return model

    def _compile_model(self, model, step: str) -> None:
        if step == "first_finetuning":
            lr = self.first_finetuning_lr
        elif step == "second_finetuning":
            lr = self.second_finetuning_lr

        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            optimizer=optimizer, loss=loss_fn, metrics=["categorical_accuracy"]
        )

    def _train_model_step(
        self,
        model: Model,
        train_data_gen: ImageDataGenerator,
        val_data_gen: ImageDataGenerator,
        step: str,
    ):
        callbacks = self._get_callbacks(step=step)
        self.logger.info(f"Training ({step})...")

        if step == "first_finetuning":
            epochs = self.first_finetuning_epochs
        elif step == "second_finetuning":
            epochs = self.second_finetuning_epochs

        history = model.fit(
            train_data_gen,
            steps_per_epoch=int(np.ceil(train_data_gen.n / float(self.batch_size))),
            epochs=epochs,
            validation_data=val_data_gen,
            validation_steps=int(np.ceil(val_data_gen.n / float(self.batch_size))),
            callbacks=callbacks,
        )
        metrics = self._get_training_metrics(history, step=step)
        self._load_best_checkpoint(model, val_data_gen, step=step)
        return metrics

    def _get_callbacks(self, step: str):
        if step == "first_finetuning":
            checkpoints_path = self.checkpoints_first_finetuning_path
        elif step == "second_finetuning":
            checkpoints_path = self.checkpoints_second_finetuning_path

        checkpoint = ModelCheckpoint(
            checkpoints_path,
            monitor="val_categorical_accuracy",
            verbose=1,
            save_best_only=True,
            mode="max",
        )
        callbacks = [checkpoint]
        return callbacks

    def _get_training_metrics(self, history, step: str) -> Dict[str, Any]:
        metrics = {
            "train_acc": history.history["categorical_accuracy"],
            "val_acc": history.history["val_categorical_accuracy"],
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
        }
        last_train_acc = history.history["categorical_accuracy"][-1]
        last_val_acc = history.history["val_categorical_accuracy"][-1]
        self.logger.info(
            f"Last epoch metrics at {step}: acc={last_train_acc:.3f}, val_acc={last_val_acc:.3f}"
        )
        return metrics

    def _load_best_checkpoint(
        self, model, val_data_gen: ImageDataGenerator, step: str
    ) -> None:
        best_acc = 0
        best_checkpoint = ""
        if step == "first_finetuning":
            checkpoints_path = "/".join(
                self.checkpoints_first_finetuning_path.split("/")[:-1]
            )
        elif step == "second_finetuning":
            checkpoints_path = "/".join(
                self.checkpoints_second_finetuning_path.split("/")[:-1]
            )
        for checkpoint in os.listdir(checkpoints_path):
            if checkpoint.startswith("."):
                continue
            checkpoint_acc = float(checkpoint[-11:-5])
            if checkpoint_acc > best_acc:
                best_loss = checkpoint_acc
                best_checkpoint = checkpoint

        model.load_weights(f"{checkpoints_path}/{best_checkpoint}")
        val_acc = model.evaluate(val_data_gen)[1]
        self.logger.info(f"Accuracy in best epoch of {step}: {val_acc}")
