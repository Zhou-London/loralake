import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .buckets import BucketPlan
from .models import AnimaModels

TomlScalar = str | int | float | bool
TomlValue = TomlScalar | list[TomlScalar]
TomlTable = Mapping[str, "TomlValue | TomlTable"]


@dataclass(frozen=True)
class TrainSettings:
    rank: int = 32
    learning_rate: float = 2e-5
    epochs: int = 10
    batch_size: int = 4
    resolution: int = 1024
    save_every_epochs: int = 2
    optimizer: str = "adamw_optimi"
    caption_dropout: float = 0.0
    gradient_checkpointing: bool = True


def _toml_value(value: TomlValue) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(value, ensure_ascii=False)


def to_toml(table: TomlTable) -> str:
    scalars = "".join(f"{key} = {_toml_value(value)}\n"
                      for key, value in table.items() if not isinstance(value, Mapping))
    sections = [f"[{key}]\n" + to_toml(value) for key, value in table.items() if isinstance(value, Mapping)]
    return "\n".join([scalars, *sections])


def dataset_config(view_dir: Path, settings: TrainSettings, buckets: BucketPlan) -> str:
    general: dict[str, TomlValue] = {
        "resolutions": [settings.resolution],
        "enable_ar_bucket": True,
        "ar_buckets": list(buckets.aspect_ratios),
    }
    directory: dict[str, TomlValue] = {"path": view_dir.as_posix(), "num_repeats": 1}
    return to_toml(general) + "\n[[directory]]\n" + to_toml(directory)


def training_config(models: AnimaModels, settings: TrainSettings, dataset_config_file: Path,
                    checkpoint_dir: Path) -> TomlTable:
    return {
        "output_dir": checkpoint_dir.as_posix(),
        "dataset": dataset_config_file.as_posix(),
        "epochs": settings.epochs,
        "micro_batch_size_per_gpu": settings.batch_size,
        "pipeline_stages": 1,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "save_every_n_epochs": settings.save_every_epochs,
        "activation_checkpointing": settings.gradient_checkpointing,
        "uncond_fraction": settings.caption_dropout,
        "save_dtype": "bfloat16",
        "caching_batch_size": settings.batch_size,
        "steps_per_print": 1,
        "model": {
            "type": "anima",
            "transformer_path": models.dit.as_posix(),
            "vae_path": models.vae.as_posix(),
            "llm_path": models.qwen3.as_posix(),
            "dtype": "bfloat16",
            "llm_adapter_lr": 0,
        },
        "adapter": {"type": "lora", "rank": settings.rank, "dtype": "bfloat16"},
        "optimizer": {
            "type": settings.optimizer,
            "lr": settings.learning_rate,
            "betas": [0.9, 0.99],
            "weight_decay": 0.01,
            "eps": 1e-8,
        },
    }
