"""llk --upload：把训练输出或数据集传到 Hugging Face 上自己的私有仓库（比如在计算中心训练完之后）。

用 config 里的 HF_API_KEY（需要 write 权限），不需要浏览器登录。仓库默认是 <用户名>/loralake（私有 dataset 仓库，
不存在时自动创建），可用 HF_UPLOAD_REPO 指定：

- 训练输出 output/<名字>/ → output/<名字>/（整个目录，中断的训练带着训练状态，可以在别处续训）
- 数据集 → 先 --pack 成 .tar.zst，再上传到 datasets/<名字>.tar.zst
"""

import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from .archive import ARCHIVE_SUFFIX, pack
from .core.config import Config
from .core.console import confirm, console
from .core.dataset import resolve_dataset
from .core.formatting import human_size

REPO_TYPE = "dataset"
DEFAULT_REPO_NAME = "loralake"


class UploadError(Exception):
    pass


def _token_role(info: dict[str, Any]) -> str:
    return ((info.get("auth") or {}).get("accessToken") or {}).get("role") or ""


def _prepare(config: Config) -> tuple[HfApi, str]:
    """检查 token，确保仓库存在，返回 (api, repo_id)。"""
    if not config.hf_token:
        raise UploadError("config 里没有 HF_API_KEY。在 https://huggingface.co/settings/tokens 生成一个 write 权限的 token 填进去")
    if config.proxy:
        os.environ.setdefault("HTTPS_PROXY", config.proxy)
        os.environ.setdefault("HTTP_PROXY", config.proxy)
    api = HfApi(token=config.hf_token)
    info = api.whoami()
    if _token_role(info) == "read":
        raise UploadError("HF_API_KEY 是只读 token，上传需要 write 权限。在 https://huggingface.co/settings/tokens 重新生成一个")
    repo_id = config.hf_upload_repo or f"{info['name']}/{DEFAULT_REPO_NAME}"
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)
    return api, repo_id


def _repo_url(repo_id: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/tree/main/{path}"


def _upload_run(api: HfApi, repo_id: str, run_dir: Path, yes: bool) -> int:
    dest = f"output/{run_dir.name}"
    size = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    if not yes and not confirm(f"上传训练输出 {run_dir}（{human_size(size)}）到 {repo_id} 的 {dest}/？"):
        return 1
    api.upload_folder(repo_id=repo_id, repo_type=REPO_TYPE, folder_path=run_dir, path_in_repo=dest,
                      commit_message=f"Upload {dest}")
    console.print(f"[green]已上传 → {_repo_url(repo_id, dest)}")
    return 0


def _upload_dataset(api: HfApi, repo_id: str, dataset_dir: Path, yes: bool, config: Config) -> int:
    archive = config.dataset_dir / f"{dataset_dir.name}{ARCHIVE_SUFFIX}"
    dest = f"datasets/{archive.name}"
    if not yes and not confirm(f"把数据集 {dataset_dir.name} 打包上传到 {repo_id} 的 {dest}？"):
        return 1
    if pack(str(dataset_dir), True, config) != 0:
        return 1
    api.upload_file(repo_id=repo_id, repo_type=REPO_TYPE, path_or_fileobj=archive, path_in_repo=dest,
                    commit_message=f"Upload {dest}")
    console.print(f"[green]已上传 → {_repo_url(repo_id, 'datasets')}")
    return 0


def upload(name: str, yes: bool, config: Config) -> int:
    run_dir = config.output_dir / name
    is_run = name not in (".", "..") and "/" not in name and run_dir.is_dir()
    dataset_dir = None if is_run else resolve_dataset(name, config.dataset_dir)
    if not is_run and dataset_dir is None:
        console.print(f"[red]找不到训练输出 {run_dir} 或数据集 {name}")
        return 1
    try:
        api, repo_id = _prepare(config)
        if dataset_dir is None:
            return _upload_run(api, repo_id, run_dir, yes)
        return _upload_dataset(api, repo_id, dataset_dir, yes, config)
    except UploadError as e:
        console.print(f"[red]{e}")
    except HfHubHTTPError as e:
        console.print(f"[red]Hugging Face 返回错误：{e}")
    except OSError as e:
        console.print(f"[red]网络错误：{e}。重新运行同一命令即可")
    except KeyboardInterrupt:
        console.print("\n[yellow]已中断，重新运行同一命令即可")
        return 130
    return 1
