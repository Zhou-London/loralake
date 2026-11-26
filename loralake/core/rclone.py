"""调用 rclone 访问 Google Drive（llk --upload 和 llk -d <Drive 链接>）。授权由 rclone 自己完成，见 README。"""

import os
import shutil
import subprocess

from .config import Config

SETUP_HINT = "安装和授权方法见 README 的「rclone 安装和授权」"


def remote_name(config: Config) -> str:
    """RCLONE_REMOTE 里的 remote 部分，如 gdrive:loralake → gdrive:"""
    return config.rclone_remote.split(":", 1)[0] + ":"


def remote_path(remote: str, *parts: str) -> str:
    base = remote.rstrip("/")
    return base + ("" if base.endswith(":") else "/") + "/".join(parts)


def _env(config: Config) -> dict[str, str]:
    env = dict(os.environ)
    if config.proxy:
        env.update(HTTP_PROXY=config.proxy, HTTPS_PROXY=config.proxy)
    return env


def check(config: Config) -> str | None:
    """rclone 可用且配置了 RCLONE_REMOTE 指向的 remote 时返回 None，否则返回错误说明。"""
    if shutil.which("rclone") is None:
        return f"找不到 rclone。{SETUP_HINT}"
    if ":" not in config.rclone_remote:
        return f"RCLONE_REMOTE 应写成 <remote 名>:<路径>，例如 gdrive:loralake（当前为 {config.rclone_remote!r}）"
    listed = capture(config, "listremotes")
    if remote_name(config) not in listed.stdout.split():
        return f"rclone 里还没有名为 {remote_name(config)} 的 remote，先运行 rclone config 添加。{SETUP_HINT}"
    return None


def run(config: Config, *args: str) -> int:
    """运行 rclone 并显示进度，返回退出码。"""
    return subprocess.run(["rclone", *args, "--progress"], env=_env(config)).returncode


def capture(config: Config, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["rclone", *args], capture_output=True, text=True, env=_env(config))
