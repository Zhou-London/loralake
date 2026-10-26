import fnmatch
from dataclasses import dataclass
from pathlib import Path

from ..core.config import Config

DIT_PATTERNS = ("anima-base*.safetensors",)
QWEN3_PATTERNS = ("qwen_3_06b*.safetensors", "qwen3_06b*.safetensors", "qwen3*0.6b*.safetensors")
VAE_PATTERNS = ("*qwen*image*vae*.safetensors", "qwen_image_vae*.safetensors")


class ModelNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class AnimaModels:
    dit: Path
    qwen3: Path
    vae: Path


def _find(models_dir: Path | None, subdir: str, patterns: tuple[str, ...], label: str, config_key: str) -> Path:
    directory = models_dir / subdir if models_dir else None
    if directory and directory.is_dir():
        matches = sorted(
            path for path in directory.iterdir()
            if path.is_file() and any(fnmatch.fnmatch(path.name.lower(), pattern) for pattern in patterns)
        )
        if matches:
            return matches[-1]
    raise ModelNotFoundError(f"找不到{label}（在 {directory} 中查找 {', '.join(patterns)}），可在 config 中用 {config_key}= 指定路径")


def _explicit(path: Path | None, label: str) -> Path | None:
    if path is not None and not path.is_file():
        raise ModelNotFoundError(f"config 中指定的{label}不存在: {path}")
    return path


def locate_anima_models(config: Config) -> AnimaModels:
    root = config.comfyui_models_dir
    return AnimaModels(
        dit=_explicit(config.anima_dit, "Anima 底模") or _find(root, "diffusion_models", DIT_PATTERNS, " Anima 底模", "ANIMA_DIT"),
        qwen3=_explicit(config.anima_qwen3, "Qwen3 文本编码器") or _find(root, "text_encoders", QWEN3_PATTERNS, " Qwen3 文本编码器", "ANIMA_QWEN3"),
        vae=_explicit(config.anima_vae, "VAE") or _find(root, "vae", VAE_PATTERNS, " Qwen-Image VAE", "ANIMA_VAE"),
    )
