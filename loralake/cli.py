import asyncio
from collections.abc import Coroutine
from pathlib import Path
from dataclasses import replace
from typing import Annotated, Any, Optional

import click
import typer

from .archive import pack, unpack
from .core.config import Config
from .core.console import console
from .download.commands import download, search
from .download.gdrive import download_drive, is_drive_url
from .preprocess.commands import preprocess
from .preprocess.review import DEFAULT_PORT, review
from .preprocess.tags import canonical_tags
from .presets import Preset, PresetError, load_preset
from .train.commands import resume, train

DEFAULT_SEARCH_LIMIT = 10
DEFAULT_PREVIEW_SAMPLES = 5
# 续训时这些选项沿用 run.json 里的值
RESUME_FIXED = frozenset({"preset_name", "trigger", "remove_tags", "exclude_images", "name", "rank", "learning_rate",
                          "epochs", "batch_size", "resolution", "save_every", "gradient_checkpointing", "dedup"})
# 从 Google Drive 下载单个文件时用不到的下载选项
DRIVE_UNUSED = frozenset({"output", "limit", "workers", "use_mirror", "include", "exclude"})

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def run_async(command: Coroutine[Any, Any, int]) -> int:
    try:
        return asyncio.run(command)
    except KeyboardInterrupt:
        console.print("\n[yellow]已中断，重新运行同一命令可继续下载")
        return 130


def given(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def tag_list(value: str | None) -> frozenset[str] | None:
    return None if value is None else canonical_tags(value.split(","))


def resolve_preset(preset_name: str | None, trigger: str | None, remove_tags: str | None, exclude_images: str | None,
                   dedup: bool | None, train_overrides: dict[str, Any]) -> Preset:
    try:
        preset = load_preset(preset_name)
    except PresetError as e:
        console.print(f"[red]{e}")
        raise typer.Exit(1)
    caption = replace(preset.caption, **given(trigger=trigger.strip() if trigger is not None else None,
                                              remove_tags=tag_list(remove_tags)))
    excluded = tag_list(exclude_images)
    return replace(preset, caption=caption, train=replace(preset.train, **train_overrides),
                   exclude_images=preset.exclude_images if excluded is None else excluded,
                   dedup=preset.dedup if dedup is None else dedup)


@app.command()
def main(
    ctx: typer.Context,
    download_tag: Annotated[Optional[str], typer.Option("--download", "-d", metavar="TAG")] = None,
    search_tag: Annotated[Optional[str], typer.Option("--search", "-s", metavar="TAG")] = None,
    preprocess_dataset: Annotated[Optional[str], typer.Option("--preprocess", "-p", metavar="DATASET")] = None,
    train_dataset: Annotated[Optional[str], typer.Option("--train", "-t", metavar="DATASET")] = None,
    resume_run: Annotated[Optional[str], typer.Option("--resume", "-r", metavar="RUN")] = None,
    review_dataset: Annotated[Optional[str], typer.Option("--review", metavar="DATASET")] = None,
    pack_dataset: Annotated[Optional[str], typer.Option("--pack", metavar="DATASET")] = None,
    unpack_archive: Annotated[Optional[str], typer.Option("--unpack", metavar="ARCHIVE")] = None,
    extra_tags: Annotated[Optional[list[str]], typer.Argument(hidden=True, metavar="TAG")] = None,
    preset_name: Annotated[Optional[str], typer.Option("--config", "-c", metavar="PRESET")] = None,
    output: Annotated[Optional[Path], typer.Option(file_okay=False)] = None,
    limit: Annotated[Optional[int], typer.Option(min=1)] = None,
    workers: Annotated[Optional[int], typer.Option(min=1)] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    use_mirror: Annotated[Optional[bool], typer.Option("--mirror/--no-mirror")] = None,
    include: Annotated[Optional[str], typer.Option(metavar="TAGS")] = None,
    exclude: Annotated[Optional[str], typer.Option(metavar="TAGS")] = None,
    trigger: Annotated[Optional[str], typer.Option()] = None,
    remove_tags: Annotated[Optional[str], typer.Option(metavar="TAGS")] = None,
    exclude_images: Annotated[Optional[str], typer.Option(metavar="TAGS")] = None,
    dedup: Annotated[Optional[bool], typer.Option("--dedup/--no-dedup")] = None,
    port: Annotated[int, typer.Option(min=1, max=65535)] = DEFAULT_PORT,
    name: Annotated[Optional[str], typer.Option()] = None,
    rank: Annotated[Optional[int], typer.Option(min=1)] = None,
    learning_rate: Annotated[Optional[float], typer.Option("--lr", min=0)] = None,
    epochs: Annotated[Optional[int], typer.Option(min=1)] = None,
    batch_size: Annotated[Optional[int], typer.Option(min=1)] = None,
    resolution: Annotated[Optional[int], typer.Option(min=256)] = None,
    save_every: Annotated[Optional[int], typer.Option(min=1, metavar="EPOCHS")] = None,
    gradient_checkpointing: Annotated[Optional[bool], typer.Option("--gradient-checkpointing/--no-gradient-checkpointing")] = None,
) -> None:
    actions = [flag for flag, value in (
        ("--download", download_tag), ("--search", search_tag),
        ("--preprocess", preprocess_dataset), ("--train", train_dataset), ("--resume", resume_run),
        ("--review", review_dataset), ("--pack", pack_dataset), ("--unpack", unpack_archive)) if value]
    if len(actions) > 1:
        raise click.UsageError(f"{' 和 '.join(actions)} 不能同时使用")

    if resume_run:
        if fixed := [f"--{param.name.replace('_', '-')}" for param in ctx.command.params
                     if param.name in RESUME_FIXED and ctx.params.get(param.name) is not None]:
            raise click.UsageError(f"--resume 沿用开始训练时的设置，不能再指定 {', '.join(fixed)}")
        raise typer.Exit(resume(resume_run, yes, Config.load()))

    if review_dataset:
        raise typer.Exit(review(review_dataset, port, Config.load()))

    if pack_dataset:
        raise typer.Exit(pack(pack_dataset, yes, Config.load()))

    if unpack_archive:
        raise typer.Exit(unpack(unpack_archive, yes, Config.load()))

    if preprocess_dataset or train_dataset:
        train_overrides = given(rank=rank, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                                resolution=resolution, save_every_epochs=save_every,
                                gradient_checkpointing=gradient_checkpointing)
        preset = resolve_preset(preset_name, trigger, remove_tags, exclude_images, dedup, train_overrides)
        if preset.source:
            console.print(f"[dim]预设: {preset.source}")
        if preprocess_dataset:
            raise typer.Exit(preprocess(preprocess_dataset, preset.exclude_images, preset.dedup, preset.caption,
                                        limit or DEFAULT_PREVIEW_SAMPLES, output, Config.load()))
        assert train_dataset is not None
        raise typer.Exit(train(train_dataset, preset.exclude_images, preset.dedup, preset.caption, preset.train,
                               preset.source, name, yes, Config.load()))

    first_tag = download_tag or search_tag
    if not first_tag:
        if extra_tags:
            raise click.UsageError("需要 --download、--search、--preprocess、--train、--review、--pack 或 --unpack")
        typer.echo(ctx.get_help())
        raise typer.Exit()

    if download_tag and is_drive_url(download_tag):
        unused = [f"--{param.name.replace('_', '-')}" for param in ctx.command.params
                  if param.name in DRIVE_UNUSED and ctx.params.get(param.name) is not None]
        if extra_tags:
            raise click.UsageError("Google Drive 链接后面不能再跟 tag（链接含 & 时要加引号）")
        if unused:
            raise click.UsageError(f"从 Google Drive 下载时不能指定 {', '.join(unused)}")
        raise typer.Exit(download_drive(download_tag, yes, Config.load()))

    config = Config.load()
    if config.has_partial_danbooru_auth:
        console.print("[yellow]Danbooru 认证需要同时填写 DANBOORU_LOGIN (用户名) 和 DANBOORU_API_KEY，当前以匿名身份访问")
    query = " ".join([first_tag, *(extra_tags or [])])
    if search_tag:
        raise typer.Exit(run_async(search(query, limit or DEFAULT_SEARCH_LIMIT, config)))
    filters = {"include": tag_list(include) or frozenset(), "exclude": tag_list(exclude) or frozenset()}
    raise typer.Exit(run_async(download(query, output, limit, workers, yes, use_mirror, config, **filters)))
