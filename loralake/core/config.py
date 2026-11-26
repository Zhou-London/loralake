import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config"


def _parse_config_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().upper()] = value.strip().strip('"').strip("'")
    return values


def _resolve_dir(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _optional_path(value: str | None) -> Path | None:
    return _resolve_dir(value) if value else None


def _comfyui_models_dir(values: dict[str, str]) -> Path | None:
    explicit = values.get("COMFY_MODELS_PATH")
    return _resolve_dir(explicit) if explicit else None


@dataclass
class Config:
    dataset_dir: Path = ROOT / "datasets"
    derived_dir: Path = ROOT / "derived"
    output_dir: Path = ROOT / "output"
    cache_dir: Path = ROOT / ".cache"
    comfyui_models_dir: Path | None = None
    anima_dit: Path | None = None
    anima_qwen3: Path | None = None
    anima_vae: Path | None = None
    danbooru_login: str = ""
    danbooru_api_key: str = ""
    hf_token: str = ""
    proxy: str = ""
    cdn_workers: int = 8
    mirror_workers: int = 32

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> Self:
        values = _parse_config_file(path)
        return cls(
            dataset_dir=_resolve_dir(values.get("DATASET_DIR") or "datasets"),
            derived_dir=_resolve_dir(values.get("DERIVED_DIR") or "derived"),
            output_dir=_resolve_dir(values.get("OUTPUT_DIR") or "output"),
            comfyui_models_dir=_comfyui_models_dir(values),
            anima_dit=_optional_path(values.get("ANIMA_DIT")),
            anima_qwen3=_optional_path(values.get("ANIMA_QWEN3")),
            anima_vae=_optional_path(values.get("ANIMA_VAE")),
            danbooru_login=values.get("DANBOORU_LOGIN", ""),
            danbooru_api_key=values.get("DANBOORU_API_KEY", ""),
            hf_token=values.get("HF_API_KEY") or os.environ.get("HF_TOKEN", ""),
            proxy=values.get("PROXY", ""),
            cdn_workers=int(values.get("DOWNLOAD_WORKERS") or 8),
            mirror_workers=int(values.get("MIRROR_WORKERS") or 32),
        )

    @property
    def has_danbooru_auth(self) -> bool:
        return bool(self.danbooru_login and self.danbooru_api_key)

    @property
    def has_partial_danbooru_auth(self) -> bool:
        return bool(self.danbooru_login) != bool(self.danbooru_api_key)
