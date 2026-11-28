import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from ..core.config import ROOT
from . import guard
from .buckets import BucketPlan
from .progress import ANSI_ESCAPE, SAVING_MODEL, SAVING_STATE, STEP_PROGRESS, TrainingProgress

DIFFUSION_PIPE_DIR = ROOT / "third_party" / "diffusion-pipe"
TRAIN_SCRIPT = "train.py"
# DeepSpeed 启动的是 guard.py，它装好 NaN 检查和日志后再运行 train.py
GUARD_SCRIPT = Path(guard.__file__).resolve()
# 用 python -m 启动，不依赖 deepspeed 脚本装在哪个 bin 目录（conda 环境和 Colab 的系统 Python 不同）
DEEPSPEED = [sys.executable, "-m", "deepspeed.launcher.runner"]
ADAPTER_FILE = "adapter_model.safetensors"
EPOCH_DIR = re.compile(r"^epoch(\d+)$")
STATE_DIR = re.compile(r"^global_step(\d+)$")
LATEST_FILE = "latest"
READ_CHUNK = 1 << 16
# diffusion-pipe 在一个 epoch 结束时先保存训练状态和 LoRA，都保存完了才打印这一行
SavedCallback = Callable[[str], None]
NEW_EPOCH = re.compile(r"Started new epoch: (\d+)")


class DiffusionPipeError(Exception):
    pass


def ensure_installed() -> None:
    if not (DIFFUSION_PIPE_DIR / TRAIN_SCRIPT).is_file():
        raise DiffusionPipeError(f"找不到 diffusion-pipe（{DIFFUSION_PIPE_DIR}），请运行 ./install.sh")
    if importlib.util.find_spec("deepspeed") is None:
        raise DiffusionPipeError(f"当前 Python（{sys.executable}）中没有 deepspeed，请先运行 ./install.sh；用 conda 安装的需先激活环境: conda activate loralake")


def commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=DIFFUSION_PIPE_DIR,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _environment() -> dict[str, str]:
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
            "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1"}


@dataclass(frozen=True)
class TrainingState:
    run: str  # checkpoints/ 下 diffusion-pipe 的运行目录名，传给 --resume_from_checkpoint
    step: int


def _state_step(path: Path) -> int | None:
    match = STATE_DIR.match(path.name)
    return int(match[1]) if match else None


def latest_training_state(checkpoint_dir: Path) -> TrainingState | None:
    states = []
    for latest in checkpoint_dir.glob(f"*/{LATEST_FILE}"):
        tag = latest.read_text(encoding="utf-8").strip()
        step = _state_step(Path(tag))
        if step is not None and (latest.parent / tag).is_dir():
            states.append((latest.stat().st_mtime, TrainingState(latest.parent.name, step)))
    return max(states, key=lambda item: item[0])[1] if states else None


def prune_training_state(checkpoint_dir: Path) -> None:
    """删掉比 latest 指向的更早的训练状态。latest 在新状态写完后才更新，所以正在写的那份不会被删。"""
    for latest in checkpoint_dir.glob(f"*/{LATEST_FILE}"):
        current = _state_step(Path(latest.read_text(encoding="utf-8").strip()))
        if current is None:
            continue
        for state in latest.parent.glob("global_step*"):
            step = _state_step(state)
            if step is not None and step < current and state.is_dir():
                shutil.rmtree(state, ignore_errors=True)


def nan_event_files(checkpoint_dir: Path) -> set[Path]:
    return set((checkpoint_dir / guard.EVENTS_DIR).glob("*.json"))


def nan_events(checkpoint_dir: Path, known: set[Path]) -> list[dict[str, Any]]:
    """guard.py 写下的、不在 known 里的 NaN 记录（文件名按步数和时间排序）；每条带上自己的路径 _file。"""
    events = []
    for path in sorted(nan_event_files(checkpoint_dir) - known):
        try:
            events.append({**json.loads(path.read_text(encoding="utf-8")), "_file": path})
        except (OSError, ValueError):
            events.append({"_file": path})
    return events


def run_training(config_file: Path, log_file: Path, buckets: BucketPlan, epochs: int, console: Console,
                 checkpoint_dir: Path, resume: TrainingState | None = None, append_log: bool = False,
                 on_saved: SavedCallback | None = None) -> int:
    """on_saved：每次保存完 LoRA 和训练状态、开始下一个 epoch 时调用，参数是保存内容的说明（如「第 4 个 epoch」）。
    训练结束时的最后一次保存不会调用，由调用方在训练进程退出后处理。"""
    command = [*DEEPSPEED, "--num_gpus=1", str(GUARD_SCRIPT), "--deepspeed", "--config", str(config_file)]
    if resume is not None:
        command += ["--resume_from_checkpoint", resume.run]
    display = TrainingProgress(console, buckets, epochs, resume.step if resume else 0)
    with log_file.open("ab" if resume or append_log else "wb") as log, display:
        process = subprocess.Popen(command, cwd=DIFFUSION_PIPE_DIR, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=_environment())
        assert process.stdout is not None
        pending = b""
        saved: str | None = None  # 这个 epoch 结束时保存了什么，还没通知 on_saved
        try:
            while chunk := os.read(process.stdout.fileno(), READ_CHUNK):
                log.write(chunk)
                log.flush()
                *lines, pending = re.split(rb"[\r\n]", pending + chunk)
                for line in lines:
                    text = line.decode("utf-8", "replace")
                    display.feed(text)
                    if STEP_PROGRESS.search(text):
                        prune_training_state(checkpoint_dir)
                    elif match := SAVING_MODEL.search(text):
                        epoch = EPOCH_DIR.match(match[1])
                        saved = f"第 {epoch[1]} 个 epoch" if epoch else match[1]
                    elif SAVING_STATE.search(text):
                        saved = saved or "训练状态"
                    elif saved and NEW_EPOCH.search(text):
                        prune_training_state(checkpoint_dir)
                        if on_saved is not None:
                            on_saved(saved)
                        saved = None
            exit_code = process.wait()
            display.finish(exit_code == 0)
            return exit_code
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise


def collect_adapters(checkpoint_dir: Path, run_dir: Path, final_epoch: int, required: bool = True) -> list[Path]:
    collected = []
    for adapter in sorted(checkpoint_dir.glob(f"*/epoch*/{ADAPTER_FILE}")):
        match = EPOCH_DIR.match(adapter.parent.name)
        if match is None:
            continue
        epoch = int(match[1])
        name = run_dir.name if epoch == final_epoch else f"{run_dir.name}-{epoch:06d}"
        target = run_dir / f"{name}.safetensors"
        shutil.move(adapter, target)
        collected.append(target)
    if not collected and required:
        raise DiffusionPipeError(f"在 {checkpoint_dir} 中没有找到训练好的 LoRA")
    return collected


def remove_training_state(checkpoint_dir: Path) -> None:
    for state in checkpoint_dir.glob("*/global_step*"):
        shutil.rmtree(state)
    for latest in checkpoint_dir.glob("*/latest"):
        latest.unlink()


def log_tail(log_file: Path, lines: int = 25) -> str:
    text = ANSI_ESCAPE.sub("", log_file.read_bytes().decode("utf-8", "replace"))
    non_empty = [line for line in re.split(r"[\r\n]+", text) if line.strip()]
    return "\n".join(non_empty[-lines:])
