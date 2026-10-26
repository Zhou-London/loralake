import os
import re
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

from ..core.config import ROOT

SD_SCRIPTS_DIR = ROOT / "third_party" / "sd-scripts"
TRAIN_SCRIPT = "anima_train_network.py"
CONVERT_MODULE = "networks.convert_anima_lora_to_comfy"
READ_CHUNK = 1 << 16
STATUS_WIDTH = 60

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP_PROGRESS = re.compile(r"steps:\s*\d+%.*?\|\s*(\d+)/(\d+)")
AVERAGE_LOSS = re.compile(r"avr_loss=([0-9.]+(?:e-?\d+)?)")
LOG_PREFIX = re.compile(r"^\s*(?:\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s+)?(?:INFO|WARNING|ERROR|DEBUG)?\s*")
LOG_SOURCE = re.compile(r"\s+\S+\.py:\d+\s*$")


class SdScriptsError(Exception):
    pass


def ensure_installed() -> None:
    if not (SD_SCRIPTS_DIR / TRAIN_SCRIPT).is_file():
        raise SdScriptsError(f"找不到 sd-scripts（{SD_SCRIPTS_DIR}），请运行: git submodule update --init")


def commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=SD_SCRIPTS_DIR,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _environment() -> dict[str, str]:
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore::SyntaxWarning"}


def _readable_status(line: str) -> str:
    text = LOG_SOURCE.sub("", LOG_PREFIX.sub("", line)).strip()
    return "" if "%|" in text else text[:STATUS_WIDTH]


def _update_progress(progress: Progress, task: TaskID, line: str) -> None:
    if match := STEP_PROGRESS.search(line):
        loss = AVERAGE_LOSS.search(line)
        progress.update(task, description="训练", completed=int(match[1]), total=int(match[2]),
                        loss=f"loss {loss[1]}" if loss else "", status="")
    elif status := _readable_status(line):
        progress.update(task, status=status)


def run_training(config_file: Path, log_file: Path, console: Console) -> int:
    command = [sys.executable, TRAIN_SCRIPT, "--config_file", str(config_file)]
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
        task = progress.add_task("准备中", total=None, loss="", status="")
        process = subprocess.Popen(command, cwd=SD_SCRIPTS_DIR, stdout=subprocess.PIPE,
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
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise


def convert_to_comfy(source: Path, target: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", CONVERT_MODULE, str(source), str(target)],
        cwd=SD_SCRIPTS_DIR, env=_environment(), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0 or not target.is_file():
        raise SdScriptsError(f"转换 {source.name} 失败:\n{(result.stdout + result.stderr)[-1500:]}")


def log_tail(log_file: Path, lines: int = 25) -> str:
    text = ANSI_ESCAPE.sub("", log_file.read_bytes().decode("utf-8", "replace"))
    non_empty = [line for line in re.split(r"[\r\n]+", text) if line.strip()]
    return "\n".join(non_empty[-lines:])
