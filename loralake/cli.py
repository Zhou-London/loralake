import asyncio
from collections.abc import Coroutine
from pathlib import Path
from dataclasses import replace
from typing import Annotated, Any, Optional

import click
import typer

from .core.config import Config
from .core.console import console
from .download.commands import download, search
from .preprocess.commands import preprocess
from .preprocess.tags import canonical_tags
from .presets import Preset, PresetError, load_preset
from .train.commands import train

DEFAULT_SEARCH_LIMIT = 10
DEFAULT_PREVIEW_SAMPLES = 5

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
                   train_overrides: dict[str, Any]) -> Preset:
    try:
        preset = load_preset(preset_name)
    except PresetError as e:
        console.print(f"[red]{e}")
        raise typer.Exit(1)
    caption = replace(preset.caption, **given(trigger=trigger.strip() if trigger is not None else None,
                                              remove_tags=tag_list(remove_tags)))
    excluded = tag_list(exclude_images)
    return replace(preset, caption=caption, train=replace(preset.train, **train_overrides),
                   exclude_images=preset.exclude_images if excluded is None else excluded)


@app.command()
def main(
    ctx: typer.Context,
    download_tag: Annotated[Optional[str], typer.Option("--download", "-d", metavar="TAG")] = None,
    search_tag: Annotated[Optional[str], typer.Option("--search", "-s", metavar="TAG")] = None,
    preprocess_dataset: Annotated[Optional[str], typer.Option("--preprocess", "-p", metavar="DATASET")] = None,
    train_dataset: Annotated[Optional[str], typer.Option("--train", "-t", metavar="DATASET")] = None,
    extra_tags: Annotated[Optional[list[str]], typer.Argument(hidden=True, metavar="TAG")] = None,
    preset_name: Annotated[Optional[str], typer.Option("--config", "-c", metavar="PRESET")] = None,
    output: Annotated[Optional[Path], typer.Option(file_okay=False)] = None,
    limit: Annotated[Optional[int], typer.Option(min=1)] = None,
    workers: Annotated[Optional[int], typer.Option(min=1)] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    use_mirror: Annotated[Optional[bool], typer.Option("--mirror/--no-mirror")] = None,
    trigger: Annotated[Optional[str], typer.Option()] = None,
    remove_tags: Annotated[Optional[str], typer.Option(metavar="TAGS")] = None,
    exclude_images: Annotated[Optional[str], typer.Option(metavar="TAGS")] = None,
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
        ("--preprocess", preprocess_dataset), ("--train", train_dataset)) if value]
    if len(actions) > 1:
        raise click.UsageError(f"{' 和 '.join(actions)} 不能同时使用")

    if preprocess_dataset or train_dataset:
        train_overrides = given(rank=rank, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                                resolution=resolution, save_every_epochs=save_every,
                                gradient_checkpointing=gradient_checkpointing)
        preset = resolve_preset(preset_name, trigger, remove_tags, exclude_images, train_overrides)
        if preset.source:
            console.print(f"[dim]预设: {preset.source}")
        if preprocess_dataset:
            raise typer.Exit(preprocess(preprocess_dataset, preset.exclude_images, preset.caption,
                                        limit or DEFAULT_PREVIEW_SAMPLES, output, Config.load()))
        assert train_dataset is not None
        raise typer.Exit(train(train_dataset, preset.exclude_images, preset.caption, preset.train,
                               preset.source, name, yes, Config.load()))

    first_tag = download_tag or search_tag
    if not first_tag:
        if extra_tags:
            raise click.UsageError("需要 --download、--search、--preprocess 或 --train")
        typer.echo(ctx.get_help())
        raise typer.Exit()

    config = Config.load()
    if config.has_partial_danbooru_auth:
        console.print("[yellow]Danbooru 认证需要同时填写 DANBOORU_LOGIN (用户名) 和 DANBOORU_API_KEY，当前以匿名身份访问")
    query = " ".join([first_tag, *(extra_tags or [])])
    if search_tag:
        raise typer.Exit(run_async(search(query, limit or DEFAULT_SEARCH_LIMIT, config)))
    raise typer.Exit(run_async(download(query, output, limit, workers, yes, use_mirror, config)))
