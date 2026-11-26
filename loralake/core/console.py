from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from rich.prompt import Confirm

console = Console()


def confirm(prompt: str) -> bool:
    try:
        return Confirm.ask(prompt, console=console)
    except EOFError:
        return False


def transfer_progress() -> Progress:
    """按字节显示进度的进度条（打包、解压）。"""
    return Progress(TextColumn("[bold]{task.description}"), BarColumn(), DownloadColumn(), TransferSpeedColumn(),
                    TimeRemainingColumn(), console=console)
