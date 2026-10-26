import importlib.util
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .models import AnimaModels

TomlValue = str | int | float | bool


@dataclass(frozen=True)
class TrainSettings:
    rank: int = 32
    learning_rate: float = 2e-5
    epochs: int = 10
    batch_size: int = 4
    resolution: int = 1024
    save_every_epochs: int = 2
    seed: int = 42
    optimizer: str = "AdamW8bit"
    caption_dropout: float = 0.0
    gradient_checkpointing: bool = True

    def estimated_steps(self, images: int) -> int:
        return math.ceil(images / self.batch_size) * self.epochs


OPTIMIZER_PACKAGES = {"bitsandbytes": "8bit", "prodigyopt": "prodigy", "lion_pytorch": "lion", "schedulefree": "schedulefree"}


def missing_optimizer_package(optimizer: str) -> str | None:
    name = optimizer.lower()
    required = next((package for package, marker in OPTIMIZER_PACKAGES.items() if marker in name), None)
    return required if required and importlib.util.find_spec(required) is None else None


def _toml_value(value: TomlValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(value, ensure_ascii=False)


def to_toml(values: Mapping[str, TomlValue]) -> str:
    return "".join(f"{key} = {_toml_value(value)}\n" for key, value in values.items())


def dataset_config(view_dir: Path, settings: TrainSettings) -> str:
    general: dict[str, TomlValue] = {
        "caption_extension": ".txt",
        "enable_bucket": True,
        "bucket_no_upscale": True,
        "caption_dropout_rate": settings.caption_dropout,
    }
    dataset: dict[str, TomlValue] = {"resolution": settings.resolution, "batch_size": settings.batch_size}
    subset: dict[str, TomlValue] = {"image_dir": view_dir.as_posix(), "num_repeats": 1}
    return "\n".join([
        "[general]\n" + to_toml(general),
        "[[datasets]]\n" + to_toml(dataset),
        "[[datasets.subsets]]\n" + to_toml(subset),
    ])


def training_config(models: AnimaModels, settings: TrainSettings, dataset_config_file: Path,
                    checkpoint_dir: Path, name: str) -> dict[str, TomlValue]:
    return {
        "pretrained_model_name_or_path": models.dit.as_posix(),
        "qwen3": models.qwen3.as_posix(),
        "vae": models.vae.as_posix(),
        "dataset_config": dataset_config_file.as_posix(),
        "output_dir": checkpoint_dir.as_posix(),
        "output_name": name,
        "save_model_as": "safetensors",
        "network_module": "networks.lora_anima",
        "network_dim": settings.rank,
        "network_alpha": float(settings.rank),
        "network_train_unet_only": True,
        "optimizer_type": settings.optimizer,
        "learning_rate": settings.learning_rate,
        "lr_scheduler": "constant",
        "max_train_epochs": settings.epochs,
        "save_every_n_epochs": settings.save_every_epochs,
        "timestep_sampling": "sigmoid",
        "mixed_precision": "bf16",
        "save_precision": "bf16",
        "gradient_checkpointing": settings.gradient_checkpointing,
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "cache_text_encoder_outputs": True,
        "qwen_image_vae_2d": True,
        "max_data_loader_n_workers": 0,
        "seed": settings.seed,
    }
