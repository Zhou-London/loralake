import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.text import Text

from ..core.config import Config
from ..core.console import confirm, console
from ..core.dataset import METADATA_FILE, available_records, dataset_name, resolve_dataset
from ..preprocess.anima import AnimaCaptionOptions, anima_caption
from ..preprocess.commands import print_selection
from ..preprocess.selection import select_records
from ..preprocess.views import materialize_view
from . import sdscripts
from .models import AnimaModels, ModelNotFoundError, locate_anima_models
from .settings import TrainSettings, dataset_config, missing_optimizer_package, to_toml, training_config

DATASET_CONFIG_FILE = "dataset.toml"
TRAIN_CONFIG_FILE = "train.toml"
MANIFEST_FILE = "run.json"
LOG_FILE = "train.log"
CHECKPOINT_DIR = "checkpoints"


def default_run_name(dataset_dir: Path) -> str:
    return f"{dataset_dir.name}_{datetime.now():%Y%m%d-%H%M}"


def _json_default(value: object) -> Any:
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value).__name__}")


def write_manifest(run_dir: Path, dataset_dir: Path, images: int, exclude_images: frozenset[str],
                   caption_options: AnimaCaptionOptions, settings: TrainSettings, models: AnimaModels,
                   preset: Path | None) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": preset,
        "dataset": dataset_dir,
        "images": images,
        "exclude_images": exclude_images,
        "caption": asdict(caption_options),
        "settings": asdict(settings),
        "models": asdict(models),
        "sd_scripts_commit": sdscripts.commit(),
    }
    (run_dir / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
                                         encoding="utf-8")


def print_plan(images: int, trigger: str, settings: TrainSettings, models: AnimaModels, run_dir: Path) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("图片", f"{images:,} 张")
    table.add_row("触发词", trigger or "[dim]无[/]")
    table.add_row("训练量", f"{settings.epochs} 个 epoch × batch {settings.batch_size}（约 {settings.estimated_steps(images):,} 步）")
    table.add_row("LoRA", f"rank {settings.rank}，学习率 {settings.learning_rate:g}，{settings.optimizer}")
    table.add_row("分辨率", f"{settings.resolution}，每 {settings.save_every_epochs} 个 epoch 保存一次"
                            + ("，梯度检查点" if settings.gradient_checkpointing else ""))
    table.add_row("底模", models.dit.name)
    table.add_row("输出", str(run_dir))
    console.print(table)


def convert_checkpoints(checkpoint_dir: Path, run_dir: Path, final_epoch: int) -> list[Path]:
    duplicate_of_final = f"{run_dir.name}-{final_epoch:06d}.safetensors"
    converted = []
    for checkpoint in sorted(checkpoint_dir.glob("*.safetensors")):
        if checkpoint.name == duplicate_of_final:
            continue
        target = run_dir / checkpoint.name
        sdscripts.convert_to_comfy(checkpoint, target)
        converted.append(target)
    return converted


def train(dataset: str, exclude_images: frozenset[str], caption_options: AnimaCaptionOptions,
          settings: TrainSettings, preset: Path | None, name: str | None, yes: bool, config: Config) -> int:
    try:
        sdscripts.ensure_installed()
        models = locate_anima_models(config)
    except (sdscripts.SdScriptsError, ModelNotFoundError) as e:
        console.print(f"[red]{e}")
        return 1
    if package := missing_optimizer_package(settings.optimizer):
        console.print(f"[red]优化器 {settings.optimizer} 需要 {package}，请在 ComfyUI 环境中安装: pip install --no-deps {package}")
        return 1
    dataset_dir = resolve_dataset(dataset, config.dataset_dir)
    if dataset_dir is None:
        console.print(f"[red]找不到数据集 {dataset}（需要包含 {METADATA_FILE}）")
        return 1

    available = available_records(dataset_dir)
    selection = select_records(available, exclude_images)
    print_selection(dataset_dir, len(available), selection)
    if not selection.records:
        console.print("[yellow]没有可用的图片")
        return 1

    run_dir = config.output_dir / dataset_name(name or default_run_name(dataset_dir))
    if run_dir.exists():
        console.print(f"[red]输出目录已存在: {run_dir}，请用 --name 指定其他名称")
        return 1
    print_plan(len(selection.records), caption_options.trigger, settings, models, run_dir)
    if not yes and not confirm("是否开始训练?"):
        console.print("已取消")
        return 0

    with console.status("正在生成训练视图 ..."):
        view = materialize_view(dataset_dir, selection.records, config.derived_dir / dataset_dir.name / "anima",
                                lambda record: anima_caption(record, caption_options))
    console.print(f"训练视图: {view.path}（更新 caption {view.captions_written:,} 个）")

    checkpoint_dir = run_dir / CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True)
    dataset_config_file = run_dir / DATASET_CONFIG_FILE
    train_config_file = run_dir / TRAIN_CONFIG_FILE
    dataset_config_file.write_text(dataset_config(view.path, settings), encoding="utf-8")
    train_config_file.write_text(
        to_toml(training_config(models, settings, dataset_config_file, checkpoint_dir, run_dir.name)), encoding="utf-8")
    write_manifest(run_dir, dataset_dir, len(selection.records), exclude_images, caption_options, settings, models, preset)

    log_file = run_dir / LOG_FILE
    try:
        exit_code = sdscripts.run_training(train_config_file, log_file, console)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]训练已中断。已保存的中间版本在 {checkpoint_dir}")
        return 130
    if exit_code != 0:
        console.print(f"[red]训练失败 (exit {exit_code})，日志: {log_file}")
        console.print(Text(sdscripts.log_tail(log_file), style="dim"))
        return exit_code

    with console.status("正在转换为 ComfyUI 格式 ..."):
        try:
            outputs = convert_checkpoints(checkpoint_dir, run_dir, settings.epochs)
        except sdscripts.SdScriptsError as e:
            console.print(f"[red]{e}")
            return 1
    console.print(f"[green]训练完成，LoRA 已保存到 {run_dir}")
    for output in outputs:
        console.print(f"  {output.name}")
    return 0
