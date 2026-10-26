from rich.console import Console
from rich.prompt import Confirm

console = Console()


def confirm(prompt: str) -> bool:
    try:
        return Confirm.ask(prompt, console=console)
    except EOFError:
        return False
