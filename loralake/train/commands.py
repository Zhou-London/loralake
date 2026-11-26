import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.text import Text

from ..core.config import Config
from ..core.console import confirm, console
from ..core.dataset import METADATA_FILE, dataset_name, resolve_dataset
from ..preprocess.anima import AnimaCaptionOptions, anima_caption
from ..preprocess.commands import load_selection
from ..preprocess.dedup import LEGACY_THRESHOLD
from ..preprocess.selection import Selection
from ..preprocess.views import materialize_view
from ..presets import LEGACY_TRAIN_KEYS, PresetError, apply_options
from . import diffpipe, guard
from .buckets import BucketPlan, image_aspect_ratio, plan_buckets
from .models import AnimaModels, ModelNotFoundError, locate_anima_models
from .settings import TrainSettings, dataset_config, to_toml, training_config

DATASET_CONFIG_FILE = "dataset.toml"
TRAIN_CONFIG_FILE = "train.toml"
MANIFEST_FILE = "run.json"
LOG_FILE = "train.log"
CHECKPOINT_DIR = "checkpoints"
# 出现 NaN 时回退到上一个训练状态重试的次数
MAX_NAN_RETRIES = 3


def default_run_name(dataset_dir: Path) -> str:
    return f"{dataset_dir.name}_{datetime.now():%Y%m%d-%H%M}"


def _json_default(value: object) -> Any:
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value).__name__}")


def write_manifest(run_dir: Path, dataset_dir: Path, images: int, exclude_images: frozenset[str], dedup: int | None,
                   caption_options: AnimaCaptionOptions, settings: TrainSettings, buckets: BucketPlan,
                   models: AnimaModels, preset: Path | None) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": preset,
        "dataset": dataset_dir,
        "images": images,
        "exclude_images": exclude_images,
        "dedup": dedup,
        "caption": asdict(caption_options),
        "settings": asdict(settings),
        "buckets": asdict(buckets),
        "models": asdict(models),
        "diffusion_pipe_commit": diffpipe.commit(),
    }
    (run_dir / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
                                         encoding="utf-8")


def print_plan(images: int, trigger: str, settings: TrainSettings, buckets: BucketPlan, models: AnimaModels,
               run_dir: Path) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("图片", f"{images:,} 张")
    table.add_row("触发词", trigger or "[dim]无[/]")
    table.add_row("训练量", f"{settings.epochs} 个 epoch × batch {settings.batch_size}"
                            f"（{buckets.steps_per_epoch * settings.epochs:,} 步）")
    table.add_row("分桶", f"{len(buckets.aspect_ratios)} 种宽高比，每个 epoch 丢弃 {buckets.dropped} 张，"
                          f"平均裁切 {buckets.mean_crop:.1%}")
    table.add_row("LoRA", f"rank {settings.rank}，学习率 {settings.learning_rate:g}，{settings.optimizer}")
    table.add_row("分辨率", f"{settings.resolution}" + ("，梯度检查点" if settings.gradient_checkpointing else ""))
    table.add_row("保存", f"每 {settings.save_every_epochs} 个 epoch 保存 LoRA 和训练状态（用于续训和 NaN 回退）")
    table.add_row("底模", models.dit.name)
    table.add_row("输出", str(run_dir))
    console.print(table)


def _plan(dataset_dir: Path, exclude_images: frozenset[str], dedup: int | None,
          settings: TrainSettings) -> tuple[Selection, BucketPlan] | None:
    selection = load_selection(dataset_dir, exclude_images, dedup)
    if not selection.records:
        console.print("[yellow]没有可用的图片")
        return None
    buckets = plan_buckets((image_aspect_ratio(dataset_dir / record.file) for record in selection.records),
                           settings.batch_size)
    if buckets.steps_per_epoch == 0:
        console.print(f"[red]图片数少于 batch 大小 {settings.batch_size}，请用 --batch-size 调小")
        return None
    return selection, buckets


def _write_configs(run_dir: Path, dataset_dir: Path, selection: Selection, caption_options: AnimaCaptionOptions,
                   settings: TrainSettings, buckets: BucketPlan, models: AnimaModels, config: Config) -> Path:
    with console.status("正在生成训练视图 ..."):
        view = materialize_view(dataset_dir, selection.records, config.derived_dir / dataset_dir.name / "anima",
                                lambda record: anima_caption(record, caption_options))
    console.print(f"训练视图: {view.path}（更新 caption {view.captions_written:,} 个）")

    checkpoint_dir = run_dir / CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset_config_file = run_dir / DATASET_CONFIG_FILE
    train_config_file = run_dir / TRAIN_CONFIG_FILE
    dataset_config_file.write_text(dataset_config(view.path, settings, buckets), encoding="utf-8")
    train_config_file.write_text(
        to_toml(training_config(models, settings, dataset_config_file, checkpoint_dir)), encoding="utf-8")
    return train_config_file


def _print_lora(run_dir: Path) -> None:
    for lora in sorted(run_dir.glob("*.safetensors")):
        console.print(f"  {lora.name}")


def _print_resume_hint(run_dir: Path, settings: TrainSettings) -> None:
    state = diffpipe.latest_training_state(run_dir / CHECKPOINT_DIR)
    if state is None:
        console.print(f"[dim]还没有保存训练状态（每 {settings.save_every_epochs} 个 epoch 保存一次），无法续训")
    else:
        console.print(f"已保存第 {state.step:,} 步的训练状态，继续训练: [bold]llk --resume {run_dir.name}")


def _print_nan_event(event: dict[str, Any]) -> None:
    step = event.get("step", "?")
    console.print(f"[yellow]第 {step} 步出现 NaN/inf 梯度，已在更新权重之前停下（权重没有被破坏）")
    if t := event.get("t"):
        console.print(f"  时间步 t: {min(t):.3f}～{max(t):.3f}，loss {event.get('loss')}")
    if images := event.get("images"):
        console.print(Text(f"  图片: {', '.join(images)}"))
    if "nonfinite_grad_params" in event:
        layers = event.get("nonfinite_grad_layers") or []
        where = f"，流水线第 {layers[0]}～{layers[-1]} 层" if layers else ""
        console.print(f"  非有限梯度: {event['nonfinite_grad_params']}/{event.get('trainable_params', '?')} 个 LoRA 参数{where}")
    replay = event.get("replay") or {}
    if "error" in replay:
        console.print(f"  [dim]重放失败: {replay['error']}")
    elif replay:
        verdict = "复现了" if replay.get("reproduced") else "没有复现（可能是不确定的计算）"
        console.print(f"  用同一 batch 重放: {verdict}；最先出现非有限值的前向输出: "
                      f"{replay.get('first_nonfinite_forward') or '无'}")
        if backward := replay.get("nonfinite_backward_order"):
            console.print(Text(f"  反向传播中最先出现非有限梯度的模块: {', '.join(backward[:3])}"))
    console.print(f"[dim]  详细信息: {event['_file']}")


def _run(run_dir: Path, train_config_file: Path, settings: TrainSettings, buckets: BucketPlan,
         resume_from: diffpipe.TrainingState | None) -> int:
    checkpoint_dir = run_dir / CHECKPOINT_DIR
    log_file = run_dir / LOG_FILE
    try:
        for attempt in range(MAX_NAN_RETRIES + 1):
            known_events = diffpipe.nan_event_files(checkpoint_dir)
            exit_code = diffpipe.run_training(train_config_file, log_file, buckets, settings.epochs, console,
                                              checkpoint_dir, resume_from, append_log=attempt > 0)
            if not (events := diffpipe.nan_events(checkpoint_dir, known_events)):
                break
            _print_nan_event(events[-1])
            if attempt == MAX_NAN_RETRIES:
                diffpipe.collect_adapters(checkpoint_dir, run_dir, settings.epochs, required=False)
                console.print(f"[red]重试 {MAX_NAN_RETRIES} 次后仍然出现 NaN，停止训练。诊断信息在 "
                              f"{checkpoint_dir / guard.EVENTS_DIR}")
                _print_lora(run_dir)
                _print_resume_hint(run_dir, settings)
                return 1
            resume_from = diffpipe.latest_training_state(checkpoint_dir)
            where = f"第 {resume_from.step:,} 步的训练状态" if resume_from else "开头（还没有保存训练状态）"
            console.print(f"[yellow]回退到{where}，换一个随机种子重试（{attempt + 1}/{MAX_NAN_RETRIES}）")
    except KeyboardInterrupt:
        diffpipe.collect_adapters(checkpoint_dir, run_dir, settings.epochs, required=False)
        console.print(f"\n[yellow]训练已中断，已保存的 LoRA 在 {run_dir}")
        _print_lora(run_dir)
        _print_resume_hint(run_dir, settings)
        return 130
    if exit_code != 0:
        diffpipe.collect_adapters(checkpoint_dir, run_dir, settings.epochs, required=False)
        console.print(f"[red]训练失败 (exit {exit_code})，日志: {log_file}")
        console.print(Text(diffpipe.log_tail(log_file), style="dim"))
        _print_resume_hint(run_dir, settings)
        return exit_code

    try:
        diffpipe.collect_adapters(checkpoint_dir, run_dir, settings.epochs)
    except diffpipe.DiffusionPipeError as e:
        console.print(f"[red]{e}")
        return 1
    diffpipe.remove_training_state(checkpoint_dir)
    console.print(f"[green]训练完成，LoRA 已保存到 {run_dir}")
    _print_lora(run_dir)
    return 0


def train(dataset: str, exclude_images: frozenset[str], dedup: int | None, caption_options: AnimaCaptionOptions,
          settings: TrainSettings, preset: Path | None, name: str | None, yes: bool, config: Config) -> int:
    try:
        diffpipe.ensure_installed()
        models = locate_anima_models(config)
    except (diffpipe.DiffusionPipeError, ModelNotFoundError) as e:
        console.print(f"[red]{e}")
        return 1
    dataset_dir = resolve_dataset(dataset, config.dataset_dir)
    if dataset_dir is None:
        console.print(f"[red]找不到数据集 {dataset}（需要包含 {METADATA_FILE}）")
        return 1

    run_dir = config.output_dir / dataset_name(name or default_run_name(dataset_dir))
    if run_dir.exists():
        console.print(f"[red]输出目录已存在: {run_dir}，请用 --name 指定其他名称；继续中断的训练用 --resume {run_dir.name}")
        return 1
    if (plan := _plan(dataset_dir, exclude_images, dedup, settings)) is None:
        return 1
    selection, buckets = plan
    print_plan(len(selection.records), caption_options.trigger, settings, buckets, models, run_dir)
    if not yes and not confirm("是否开始训练?"):
        console.print("已取消")
        return 0

    run_dir.mkdir(parents=True)
    train_config_file = _write_configs(run_dir, dataset_dir, selection, caption_options, settings, buckets, models,
                                       config)
    write_manifest(run_dir, dataset_dir, len(selection.records), exclude_images, dedup, caption_options, settings,
                   buckets, models, preset)
    return _run(run_dir, train_config_file, settings, buckets, None)


def _resolve_run(run: str, output_dir: Path) -> Path | None:
    candidates = [Path(run), output_dir / run, output_dir / dataset_name(run)]
    return next((path for path in candidates if (path / MANIFEST_FILE).is_file()), None)


def _json_equal(value: object, stored: object) -> bool:
    return json.loads(json.dumps(value, default=_json_default)) == stored


def _current_settings(values: dict[str, Any]) -> dict[str, Any]:
    """去掉旧版 run.json 里已经不用的设置。"""
    return {key: value for key, value in values.items() if key not in LEGACY_TRAIN_KEYS}


def _recorded_dedup(value: object) -> int | None:
    """run.json 里的 dedup：阈值，null/false 表示没去重；旧版本记录的 true 对应当时固定的阈值。"""
    if value is True:
        return LEGACY_THRESHOLD
    if value is None or value is False:
        return None
    if not isinstance(value, int):
        raise TypeError(f"dedup 的值 {value!r} 无效")
    return value


def resume(run: str, yes: bool, config: Config) -> int:
    try:
        diffpipe.ensure_installed()
        models = locate_anima_models(config)
    except (diffpipe.DiffusionPipeError, ModelNotFoundError) as e:
        console.print(f"[red]{e}")
        return 1
    run_dir = _resolve_run(run, config.output_dir)
    if run_dir is None:
        console.print(f"[red]找不到训练输出 {run}（{config.output_dir} 下需要包含 {MANIFEST_FILE} 的目录）")
        return 1
    state = diffpipe.latest_training_state(run_dir / CHECKPOINT_DIR)
    if state is None:
        if (run_dir / f"{run_dir.name}.safetensors").exists():
            console.print(f"[green]{run_dir.name} 已经训练完成")
            _print_lora(run_dir)
            return 0
        console.print(f"[red]{run_dir} 中没有保存的训练状态，无法续训")
        return 1

    try:
        manifest = json.loads((run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        settings = apply_options(TrainSettings(), _current_settings(manifest["settings"]), "settings")
        caption_options = apply_options(AnimaCaptionOptions(), manifest["caption"], "caption")
        exclude_images = frozenset(manifest["exclude_images"])
        dedup = _recorded_dedup(manifest.get("dedup"))
        recorded_dataset = Path(manifest["dataset"])
        recorded_dit = Path(manifest["models"]["dit"]).name
        recorded_buckets = manifest["buckets"]
        preset = manifest.get("preset")
    except (OSError, ValueError, KeyError, TypeError, PresetError) as e:
        console.print(f"[red]无法读取 {run_dir / MANIFEST_FILE}: {e}")
        return 1
    if preset:
        console.print(f"[dim]预设: {preset}（沿用开始训练时的设置）")
    if models.dit.name != recorded_dit:
        console.print(f"[red]底模与开始训练时不同（原来 {recorded_dit}，现在 {models.dit.name}），无法续训")
        return 1
    dataset_dir = (recorded_dataset if (recorded_dataset / METADATA_FILE).exists()
                   else resolve_dataset(recorded_dataset.name, config.dataset_dir))
    if dataset_dir is None:
        console.print(f"[red]找不到数据集 {recorded_dataset}")
        return 1

    if (plan := _plan(dataset_dir, exclude_images, dedup, settings)) is None:
        return 1
    selection, buckets = plan
    if not _json_equal(asdict(buckets), recorded_buckets):
        console.print(f"[red]数据集与开始训练时不同（原来选用 {manifest.get('images', '?')} 张，"
                      f"现在 {len(selection.records):,} 张），无法续训")
        return 1
    print_plan(len(selection.records), caption_options.trigger, settings, buckets, models, run_dir)
    total_steps = buckets.steps_per_epoch * settings.epochs
    console.print(f"从第 {state.step:,} / {total_steps:,} 步继续")
    if not yes and not confirm("是否继续训练?"):
        console.print("已取消")
        return 0

    train_config_file = _write_configs(run_dir, dataset_dir, selection, caption_options, settings, buckets, models,
                                       config)
    return _run(run_dir, train_config_file, settings, buckets, state)
