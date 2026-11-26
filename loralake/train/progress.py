"""训练时终端里的进度显示：把 diffusion-pipe 的输出分成几个阶段，每个阶段一行。

  ✓ 启动                                              0:00:12
  ✓ 扫描数据集   ━━━━━━━━━━━━━━━━━━━━━━                0:00:01
  ⠋ 编码图片     ━━━━━━━━━━╸━━━━━━━━━━━  64/139 张     0:00:21  剩余 0:00:20  宽高比 0.703（864×1216）
    ……
  ⠋ 训练         ━━━━━━━━━━━╸━━━━━━━━━━  55/160 步     0:03:40  剩余 0:07:10  epoch 7/20 · loss 0.1185 · 梯度 0.42

diffusion-pipe 编码 latent / caption 时的 tqdm 只按单个宽高比桶里的 batch 计数，这里结合 BucketPlan 里
每个桶的图片数，换算成全部图片的总进度。
"""

import re
from dataclasses import dataclass

from rich.console import Console
from rich.progress import (BarColumn, Progress, ProgressColumn, SpinnerColumn, Task, TaskID, TextColumn,
                           TimeElapsedColumn)
from rich.text import Text

from . import guard
from .buckets import BucketPlan

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP_PROGRESS = re.compile(r"steps: (\d+) loss: (\S+)")
TQDM_PROGRESS = re.compile(r"(\d+)/(\d+) \[")
GUARD_STEP = re.compile(re.escape(guard.LOG_PREFIX) + r" step (\d+) (?:epoch (\d+) )?.*?grad_norm (\S+)")
CACHE_START = re.compile(r"caching (latents|text embeddings): (?![\[(])")
CACHE_BUCKET = re.compile(r"caching (latents|text embeddings): [\[(]([0-9.]+),(?: (\d+), (\d+),)?")
CACHE_LENGTH = re.compile(r"\[CACHE\] Existing cache length: (\d+)")
DATASET_START = re.compile(r"Grouped metadata is not cached|Found grouped metadata cache")
SAVING_MODEL = re.compile(r"Saving model to directory (\S+)")
SAVING_STATE = re.compile(r"Checkpoint (global_step\d+) is about to be saved")
LOG_PREFIX = re.compile(r"^\s*(?:\[[^]]*\]\s*)*(?:\[?(?:INFO|WARNING|ERROR|DEBUG)\]?\s*)?(?:\[[^]]*\]\s*)*")
STATUS_WIDTH = 50

LAUNCH, SCAN, LATENTS, CAPTIONS, LOADING, TRAINING = "启动", "扫描数据集", "编码图片", "编码 caption", "加载模型", "训练"
CACHE_PHASES = {"latents": LATENTS, "text embeddings": CAPTIONS}


class RemainingColumn(ProgressColumn):
    """只给知道总量、还没完成的阶段显示剩余时间。"""

    def render(self, task: Task) -> Text:
        if task.finished or task.total is None or task.time_remaining is None:
            return Text("")
        minutes, seconds = divmod(int(task.time_remaining), 60)
        hours, minutes = divmod(minutes, 60)
        return Text(f"剩余 {hours}:{minutes:02d}:{seconds:02d}", style="progress.remaining")


def readable_status(line: str) -> str:
    """普通日志行去掉时间戳和级别前缀后的内容；进度条和每步的诊断行返回空。"""
    if guard.LOG_PREFIX in line:
        return ""
    text = LOG_PREFIX.sub("", line).strip()
    return "" if "%|" in text or "it/s]" in text or "s/it]" in text else text[:STATUS_WIDTH]


@dataclass
class _CacheBucket:
    size: int  # 这个桶的图片数，不知道时为 0
    fraction: float = 0.0


class TrainingProgress:
    """读 diffusion-pipe 的输出行，更新各阶段的进度条。"""

    def __init__(self, console: Console, buckets: BucketPlan, epochs: int, resume_step: int = 0) -> None:
        self.total_steps = buckets.steps_per_epoch * epochs
        self.steps_per_epoch = buckets.steps_per_epoch
        self.epochs = epochs
        self.resume_step = resume_step
        self.images = sum(buckets.images)
        self.bucket_sizes = {round(ratio, 3): count for ratio, count in zip(buckets.aspect_ratios, buckets.images)}
        self.progress = Progress(
            SpinnerColumn(finished_text="[green]✓"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=28),
            TextColumn("{task.fields[count]}"),
            TimeElapsedColumn(),
            RemainingColumn(),
            TextColumn("[dim]{task.fields[status]}"),
            console=console,
        )
        self.tasks: dict[str, TaskID] = {}
        self.current: str | None = None
        self.done: set[str] = set()
        # 正在编码的阶段：已完成的桶的图片数，和当前桶
        self.cached_images = 0
        self.bucket: _CacheBucket | None = None
        self.bucket_ratio: float | None = None
        self.loss = ""
        self.grad_norm = ""
        self.epoch: int | None = None  # diffusion-pipe 数据加载器的 epoch（guard 每步打印）

    def __enter__(self) -> "TrainingProgress":
        self.progress.__enter__()
        self._start(LAUNCH)
        return self

    def __exit__(self, *exc: object) -> None:
        self.progress.__exit__(*exc)

    # ---- 阶段 ----

    def _start(self, phase: str) -> None:
        if phase == self.current or phase in self.done:
            return
        if self.current is not None:
            self._finish(self.current)
        self.current = phase
        if phase in (LATENTS, CAPTIONS):
            self.cached_images, self.bucket, self.bucket_ratio = 0, None, None
            self.tasks[phase] = self.progress.add_task(phase, total=self.images or None, count=self._count(0, "张"),
                                                       status="")
        elif phase == TRAINING:
            self.tasks[phase] = self.progress.add_task(phase, total=self.total_steps, completed=self.resume_step,
                                                       count=self._count(self.resume_step, "步"), status="")
        else:
            self.tasks[phase] = self.progress.add_task(phase, total=None, count="", status="")

    def _finish(self, phase: str) -> None:
        task_id = self.tasks[phase]
        task = self.progress.tasks[task_id]
        if task.total is None:
            self.progress.update(task_id, total=1, completed=1, status="")
        elif phase in (LATENTS, CAPTIONS):
            self.progress.update(task_id, completed=task.total, count=self._count(int(task.total), "张"), status="")
        elif phase == TRAINING:
            self.progress.update(task_id, status=self._training_status())
        self.done.add(phase)

    def finish(self, success: bool) -> None:
        """训练进程退出后调用：成功时把最后一个阶段标成完成。"""
        if success and self.current is not None and self.current not in self.done:
            if self.current == TRAINING:
                task_id = self.tasks[TRAINING]
                completed = int(self.progress.tasks[task_id].completed)
                self.total_steps = completed
                self.progress.update(task_id, total=completed, count=self._count(completed, "步"))
            self._finish(self.current)

    def _count(self, done: int, unit: str) -> str:
        if unit == "步":
            return f"{done:,}/{self.total_steps:,} 步"
        return f"{done:,}/{self.images:,} 张" if self.images else ""

    # ---- 读取一行输出 ----

    def feed(self, raw_line: str) -> None:
        line = ANSI_ESCAPE.sub("", raw_line)
        if match := STEP_PROGRESS.search(line):
            self._on_step(int(match[1]), match[2])
        elif match := GUARD_STEP.search(line):
            if match[2]:
                self.epoch = int(match[2])
            self.grad_norm = match[3]
        elif DATASET_START.search(line):
            # 第一次出现是扫描数据集；编码完 latent 之后 diffusion-pipe 会再读一遍缓存，那时已经在准备训练
            self._start(LOADING if LATENTS in self.done or self.current == LATENTS else SCAN)
        elif match := CACHE_START.search(line):
            phase = CACHE_PHASES[match[1]]
            self._start(phase if self._before_loading() else LOADING)
        elif match := CACHE_BUCKET.search(line):
            if self.current in (LATENTS, CAPTIONS):
                self._on_bucket(float(match[2]), match[3], match[4])
        elif match := CACHE_LENGTH.search(line):
            if self.current in (LATENTS, CAPTIONS) and self.bucket and self.bucket.size:
                self._on_bucket_fraction(min(int(match[1]) / self.bucket.size, 1.0))
        elif match := SAVING_MODEL.search(line):
            self._training_note(f"保存 LoRA {match[1]}")
        elif match := SAVING_STATE.search(line):
            self._training_note(f"保存训练状态 {match[1]}")
        elif match := TQDM_PROGRESS.search(line):
            self._on_tqdm(int(match[1]), int(match[2]), line)
        elif status := readable_status(line):
            if self.current in (LAUNCH, LOADING, SCAN):
                self.progress.update(self.tasks[self.current], status=status)

    def _before_loading(self) -> bool:
        return LOADING not in self.done and self.current != LOADING and CAPTIONS not in self.done

    def _on_bucket(self, ratio: float, width: str | None, height: str | None) -> None:
        if ratio != self.bucket_ratio:
            if self.bucket is not None:
                self.cached_images += self.bucket.size
            self.bucket_ratio = ratio
            self.bucket = _CacheBucket(self._bucket_size(ratio))
        where = f"宽高比 {ratio:g}" + (f"（{width}×{height}）" if width and height else "")
        self._update_cache(status=where)

    def _bucket_size(self, ratio: float) -> int:
        return next((count for key, count in self.bucket_sizes.items() if abs(key - ratio) < 1e-3), 0)

    def _on_bucket_fraction(self, fraction: float) -> None:
        if self.bucket is not None:
            self.bucket.fraction = max(self.bucket.fraction, fraction)
            self._update_cache()

    def _update_cache(self, status: str | None = None) -> None:
        bucket = self.bucket
        done = self.cached_images + (round(bucket.size * bucket.fraction) if bucket else 0)
        done = min(done, self.images)
        fields = {"status": status} if status is not None else {}
        self.progress.update(self.tasks[self.current], completed=done, count=self._count(done, "张"), **fields)

    def _on_tqdm(self, done: int, total: int, line: str) -> None:
        if self.current in (LATENTS, CAPTIONS):
            # 编码时的 tqdm 没有前缀，按 batch 计数；Map / Adding size bucket 等带前缀的是准备工作，跳过
            if total and not line.lstrip().split("%")[0].strip(" 0123456789"):
                self._on_bucket_fraction(done / total)
        elif self.current == SCAN:
            # 扫描时有好几个小进度条（列文件、读图片尺寸、分组、保存），只显示正在做哪一步
            head = line.split("|")[0]
            label = head.rsplit(":", 1)[0].strip() if ":" in head else "列出文件"
            self.progress.update(self.tasks[SCAN], status=f"{label[:STATUS_WIDTH - 12]} {done:,}/{total:,}")

    def _on_step(self, step: int, loss: str) -> None:
        self._start(TRAINING)
        self.loss = loss
        self.progress.update(self.tasks[TRAINING], completed=step, count=self._count(step, "步"),
                             status=self._training_status())

    def _training_status(self) -> str:
        parts = []
        epoch = self.epoch
        if epoch is None and self.steps_per_epoch:  # 旧日志没有 epoch，按计划的每 epoch 步数估算
            completed = int(self.progress.tasks[self.tasks[TRAINING]].completed)
            epoch = min(max(-(-completed // self.steps_per_epoch), 1), self.epochs)
        if epoch is not None:
            parts.append(f"epoch {epoch}/{self.epochs}")
        if self.loss:
            parts.append(f"loss {self.loss}")
        if self.grad_norm:
            parts.append(f"梯度 {self.grad_norm}")
        return " · ".join(parts)

    def _training_note(self, note: str) -> None:
        if self.current == TRAINING:
            self.progress.update(self.tasks[TRAINING], status=f"{self._training_status()} · {note}")
