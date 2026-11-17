import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

from ..core.config import ROOT

DIFFUSION_PIPE_DIR = ROOT / "third_party" / "diffusion-pipe"
TRAIN_SCRIPT = "train.py"
# 用 python -m 启动，不依赖 deepspeed 脚本装在哪个 bin 目录（conda 环境和 Colab 的系统 Python 不同）
DEEPSPEED = [sys.executable, "-m", "deepspeed.launcher.runner"]
ADAPTER_FILE = "adapter_model.safetensors"
EPOCH_DIR = re.compile(r"^epoch(\d+)$")
READ_CHUNK = 1 << 16
STATUS_WIDTH = 60

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP_PROGRESS = re.compile(r"steps: (\d+) loss: ([0-9.]+(?:e-?\d+)?)")
LOG_PREFIX = re.compile(r"^\s*(?:\[[^]]*\]\s*)*(?:\[?(?:INFO|WARNING|ERROR|DEBUG)\]?\s*)?(?:\[[^]]*\]\s*)*")


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


def _readable_status(line: str) -> str:
    text = LOG_PREFIX.sub("", line).strip()
    return "" if "%|" in text or "it/s]" in text else text[:STATUS_WIDTH]


def _update_progress(progress: Progress, task: TaskID, line: str) -> None:
    if match := STEP_PROGRESS.search(line):
        progress.update(task, description="训练", completed=int(match[1]), loss=f"loss {match[2]}", status="")
    elif status := _readable_status(line):
        progress.update(task, status=status)


def run_training(config_file: Path, log_file: Path, total_steps: int, console: Console) -> int:
    command = [*DEEPSPEED, "--num_gpus=1", TRAIN_SCRIPT, "--deepspeed", "--config", str(config_file)]
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[loss]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[status]}"),
        console=console,
    )
    with log_file.open("wb") as log, progress:
        task = progress.add_task("准备中", total=total_steps, loss="", status="")
        process = subprocess.Popen(command, cwd=DIFFUSION_PIPE_DIR, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=_environment())
        assert process.stdout is not None
        pending = b""
        try:
            while chunk := os.read(process.stdout.fileno(), READ_CHUNK):
                log.write(chunk)
                log.flush()
                *lines, pending = re.split(rb"[\r\n]", pending + chunk)
                for line in lines:
                    _update_progress(progress, task, ANSI_ESCAPE.sub("", line.decode("utf-8", "replace")))
            exit_code = process.wait()
            if exit_code == 0:
                progress.update(task, total=progress.tasks[0].completed)
            return exit_code
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise


def collect_adapters(checkpoint_dir: Path, run_dir: Path, final_epoch: int) -> list[Path]:
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
    if not collected:
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
