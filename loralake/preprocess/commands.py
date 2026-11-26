import random
from pathlib import Path

from rich.text import Text

from ..core.config import Config
from ..core.console import console
from ..core.dataset import METADATA_FILE, available_records, resolve_dataset
from .anima import AnimaCaptionOptions, anima_caption
from .selection import Selection, select_and_dedup
from .views import materialize_view


def print_selection(dataset_dir: Path, available: int, selection: Selection) -> None:
    console.print(f"数据集: {dataset_dir} (共 {available:,} 张，选用 {len(selection.records):,} 张)")
    if selection.excluded_count:
        hits = "，".join(f"{tag} {count:,}" for tag, count in sorted(selection.hits_by_tag.items(), key=lambda kv: -kv[1]))
        console.print(f"排除 {selection.excluded_count:,} 张（命中 {hits}）")
    if selection.dropped_count or selection.edited_count:
        console.print(f"人工整理: 丢弃 {selection.dropped_count:,} 张，改过 tag 的 {selection.edited_count:,} 张")
    if (dedup := selection.dedup) is None:
        return
    if dedup.auto_dropped:
        console.print(f"去重: 像素相同自动去除 {len(dedup.auto_dropped):,} 张")
    if pending := dedup.pending:
        images = sum(len(group.keys) for group in pending)
        console.print(f"[yellow]{len(pending):,} 组疑似重复（{images:,} 张）待确认，确认前全部保留: "
                      f"llk --review {dataset_dir.name}")


def load_selection(dataset_dir: Path, exclude_images: frozenset[str], dedup: int | None) -> Selection:
    available = available_records(dataset_dir)
    with console.status("正在查重 ..."):
        selection = select_and_dedup(dataset_dir, available, exclude_images, dedup)
    print_selection(dataset_dir, len(available), selection)
    return selection


def preprocess(dataset: str, exclude_images: frozenset[str], dedup: int | None, caption_options: AnimaCaptionOptions,
               samples: int, output: Path | None, config: Config) -> int:
    dataset_dir = resolve_dataset(dataset, config.dataset_dir)
    if dataset_dir is None:
        console.print(f"[red]找不到数据集 {dataset}（需要包含 {METADATA_FILE}）")
        return 1
    selection = load_selection(dataset_dir, exclude_images, dedup)
    if not selection.records:
        console.print("[yellow]没有可用的图片")
        return 1

    console.print()
    for record in random.sample(selection.records, min(samples, len(selection.records))):
        console.print(Text(record.file, style="bold"))
        console.print(Text(anima_caption(record, caption_options)))
        console.print()
    if missing_year := sum(record.year is None for record in selection.records):
        console.print(f"[yellow]{missing_year:,} 张缺少发布时间（旧版下载器的数据），不会生成年份 tag")

    if output is None:
        view_dir = config.derived_dir / dataset_dir.name / "anima"
        console.print(f"[dim]以上为预览。生成训练视图: llk --preprocess {dataset} --output {view_dir}")
        return 0
    with console.status("正在生成视图 ..."):
        summary = materialize_view(dataset_dir, selection.records, output,
                                   lambda record: anima_caption(record, caption_options))
    console.print(f"[green]视图已生成: {summary.path}")
    console.print(f"  {summary.images:,} 张图片（新建硬链接 {summary.linked:,}，复制 {summary.copied:,}），"
                  f"更新 caption {summary.captions_written:,} 个，清理过期文件 {summary.removed:,} 个")
    return 0
