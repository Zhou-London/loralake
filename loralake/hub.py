"""llk --upload / --fetch：在 Hugging Face 上自己的私有仓库和本机之间传训练输出和数据集。

用 config 里的 HF_API_KEY，不需要浏览器登录（上传需要 write 权限）。仓库默认是 <用户名>/loralake
（私有 dataset 仓库，上传时不存在会自动创建），可用 HF_REPO 指定。仓库里的布局：

- output/<名字>/            训练输出整个目录（中断的训练带着训练状态，可以在别处续训）
- datasets/<名字>.tar.zst   数据集，上传前先 --pack，取回后自动 --unpack
"""

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, RepoFile, RepoFolder
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

from .archive import ARCHIVE_SUFFIX, pack, unpack
from .core.config import Config
from .core.console import confirm, console
from .core.dataset import resolve_dataset
from .core.formatting import human_size

REPO_TYPE = "dataset"
DEFAULT_REPO_NAME = "loralake"
RUNS_DIR = "output"
DATASETS_DIR = "datasets"
TOKEN_URL = "https://huggingface.co/settings/tokens"
# 下载到一半的文件放在这里，重新运行同一命令时 huggingface_hub 会从断点继续
STAGING = ".hf-fetch"


class HubError(Exception):
    pass


def _token_role(info: dict[str, Any]) -> str:
    return ((info.get("auth") or {}).get("accessToken") or {}).get("role") or ""


def _connect(config: Config, write: bool) -> tuple[HfApi, str]:
    """检查 token，返回 (api, repo_id)。"""
    if not config.hf_token:
        kind = "write" if write else "read 或 write"
        raise HubError(f"config 里没有 HF_API_KEY。在 {TOKEN_URL} 生成一个 {kind} 权限的 token 填进去")
    if config.proxy:
        os.environ.setdefault("HTTPS_PROXY", config.proxy)
        os.environ.setdefault("HTTP_PROXY", config.proxy)
    api = HfApi(token=config.hf_token)
    if config.hf_repo and not write:
        return api, config.hf_repo
    info = api.whoami()
    if write and _token_role(info) == "read":
        raise HubError(f"HF_API_KEY 是只读 token，上传需要 write 权限。在 {TOKEN_URL} 重新生成一个")
    return api, config.hf_repo or f"{info['name']}/{DEFAULT_REPO_NAME}"


def _repo_url(repo_id: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/tree/main/{path}"


def _run_hub(action: Callable[[], int]) -> int:
    try:
        return action()
    except HubError as e:
        console.print(f"[red]{e}")
    except RepositoryNotFoundError:
        console.print("[red]仓库不存在，或者 HF_API_KEY 没有访问权限。先用 llk --upload 上传过才会创建仓库")
    except HfHubHTTPError as e:
        console.print(f"[red]Hugging Face 返回错误：{e}")
    except OSError as e:
        console.print(f"[red]网络错误：{e}。重新运行同一命令即可")
    except KeyboardInterrupt:
        console.print("\n[yellow]已中断，重新运行同一命令即可")
        return 130
    return 1


# ---- 上传 ----

def _upload_run(api: HfApi, repo_id: str, run_dir: Path, yes: bool) -> int:
    dest = f"{RUNS_DIR}/{run_dir.name}"
    size = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    if not yes and not confirm(f"上传训练输出 {run_dir}（{human_size(size)}）到 {repo_id} 的 {dest}/？"):
        return 1
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type=REPO_TYPE, folder_path=run_dir, path_in_repo=dest,
                      commit_message=f"Upload {dest}")
    console.print(f"[green]已上传 → {_repo_url(repo_id, dest)}")
    return 0


def _upload_dataset(api: HfApi, repo_id: str, dataset_dir: Path, yes: bool, config: Config) -> int:
    archive = config.dataset_dir / f"{dataset_dir.name}{ARCHIVE_SUFFIX}"
    dest = f"{DATASETS_DIR}/{archive.name}"
    if not yes and not confirm(f"把数据集 {dataset_dir.name} 打包上传到 {repo_id} 的 {dest}？"):
        return 1
    if pack(str(dataset_dir), True, config) != 0:
        return 1
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)
    api.upload_file(repo_id=repo_id, repo_type=REPO_TYPE, path_or_fileobj=archive, path_in_repo=dest,
                    commit_message=f"Upload {dest}")
    console.print(f"[green]已上传 → {_repo_url(repo_id, DATASETS_DIR)}")
    return 0


def upload(name: str, yes: bool, config: Config) -> int:
    run_dir = config.output_dir / name
    is_run = name not in (".", "..") and "/" not in name and run_dir.is_dir()
    dataset_dir = None if is_run else resolve_dataset(name, config.dataset_dir)
    if not is_run and dataset_dir is None:
        console.print(f"[red]找不到训练输出 {run_dir} 或数据集 {name}")
        return 1

    def action() -> int:
        api, repo_id = _connect(config, write=True)
        if dataset_dir is None:
            return _upload_run(api, repo_id, run_dir, yes)
        return _upload_dataset(api, repo_id, dataset_dir, yes, config)

    return _run_hub(action)


# ---- 取回 ----

def _list(api: HfApi, repo_id: str, folder: str) -> list[RepoFile | RepoFolder]:
    try:
        return list(api.list_repo_tree(repo_id, folder, repo_type=REPO_TYPE))
    except EntryNotFoundError:
        return []


def _remote_names(api: HfApi, repo_id: str) -> tuple[set[str], set[str]]:
    """仓库里的训练输出名和数据集名。"""
    runs = {Path(entry.path).name for entry in _list(api, repo_id, RUNS_DIR) if isinstance(entry, RepoFolder)}
    datasets = {Path(entry.path).name.removesuffix(ARCHIVE_SUFFIX) for entry in _list(api, repo_id, DATASETS_DIR)
                if isinstance(entry, RepoFile) and entry.path.endswith(ARCHIVE_SUFFIX)}
    return runs, datasets


def _print_available(repo_id: str, runs: set[str], datasets: set[str]) -> None:
    if not runs and not datasets:
        console.print(f"[dim]{repo_id} 里还没有上传过的内容")
        return
    if runs:
        console.print(f"[dim]训练输出: {', '.join(sorted(runs))}")
    if datasets:
        console.print(f"[dim]数据集: {', '.join(sorted(datasets))}")


def _replace(source: Path, target: Path) -> None:
    """用 source 整个替换 target；中途失败时恢复原来的 target。"""
    if not target.exists():
        source.rename(target)
        return
    previous = target.with_name(f".{target.name}.previous")
    shutil.rmtree(previous, ignore_errors=True)
    target.rename(previous)
    try:
        source.rename(target)
    except BaseException:
        previous.rename(target)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def _fetch_run(api: HfApi, repo_id: str, name: str, yes: bool, config: Config) -> int:
    target = config.output_dir / name
    if target.exists() and not yes and not confirm(f"{target} 已存在，用仓库里的版本替换？"):
        return 1
    staging = config.output_dir / STAGING
    api.snapshot_download(repo_id, repo_type=REPO_TYPE, allow_patterns=f"{RUNS_DIR}/{name}/**", local_dir=staging)
    _replace(staging / RUNS_DIR / name, target)
    shutil.rmtree(staging, ignore_errors=True)
    console.print(f"[green]已下载 → {target}")
    return 0


def _fetch_dataset(api: HfApi, repo_id: str, name: str, yes: bool, config: Config) -> int:
    archive = config.dataset_dir / f"{name}{ARCHIVE_SUFFIX}"
    if archive.exists() and not yes and not confirm(f"{archive} 已存在，覆盖？"):
        return 1
    staging = config.dataset_dir / STAGING
    downloaded = api.hf_hub_download(repo_id, f"{DATASETS_DIR}/{archive.name}", repo_type=REPO_TYPE,
                                     local_dir=staging)
    os.replace(downloaded, archive)
    shutil.rmtree(staging, ignore_errors=True)
    console.print(f"[green]已下载 → {archive}")
    return unpack(str(archive), yes, config)


def fetch(name: str, yes: bool, config: Config) -> int:
    name = name.strip().removesuffix(ARCHIVE_SUFFIX)
    if not name or name in (".", "..") or "/" in name:
        console.print(f"[red]无效的名字：{name}")
        return 1

    def action() -> int:
        api, repo_id = _connect(config, write=False)
        runs, datasets = _remote_names(api, repo_id)
        if name in runs:
            return _fetch_run(api, repo_id, name, yes, config)
        if name in datasets:
            return _fetch_dataset(api, repo_id, name, yes, config)
        console.print(f"[red]{repo_id} 里没有训练输出或数据集 {name}")
        _print_available(repo_id, runs, datasets)
        return 1

    return _run_hub(action)
