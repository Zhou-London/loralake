"""llk --upload / --fetch：在 Hugging Face 上自己的私有仓库和本机之间传训练输出和数据集。

用 config 里的 HF_API_KEY，不需要浏览器登录（上传需要 write 权限）。仓库默认是 <用户名>/loralake
（私有 dataset 仓库，上传时不存在会自动创建），可用 HF_REPO 指定。仓库里的布局：

- output/<名字>/            训练输出整个目录（中断的训练带着训练状态，可以在别处续训）。上传时让仓库里的
                            这个目录和本地一致：本地已经删掉的文件（旧的训练状态等）在仓库里也删掉
- datasets/<名字>.tar.zst   数据集，上传前先 --pack，取回后自动 --unpack
"""

import os
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, RepoFile, RepoFolder
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.utils import disable_progress_bars, logging as hub_logging
from rich.table import Table

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
# 训练中自动上传时，先把训练输出的快照放在这里再上传，见 AutoUploader
UPLOAD_STAGING = ".hf-upload"
# 快照里小于这个大小的文件复制（train.log 等还在追加），更大的用硬链接（LoRA、训练状态写完就不再改）
SNAPSHOT_COPY_LIMIT = 64 << 20


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

def _folder_size(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def _push_run(api: HfApi, repo_id: str, folder: Path, name: str, message: str) -> str:
    """把 folder 上传成仓库里的 output/<name>/，仓库里这个目录下本地没有的文件删掉；没有变化的文件不重复传。"""
    dest = f"{RUNS_DIR}/{name}"
    api.upload_folder(repo_id=repo_id, repo_type=REPO_TYPE, folder_path=folder, path_in_repo=dest,
                      delete_patterns="*", commit_message=message)
    return _repo_url(repo_id, dest)


def _upload_run(api: HfApi, repo_id: str, run_dir: Path, yes: bool) -> int:
    dest = f"{RUNS_DIR}/{run_dir.name}"
    if not yes and not confirm(f"上传训练输出 {run_dir}（{human_size(_folder_size(run_dir))}）到 {repo_id} 的 {dest}/？"):
        return 1
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)
    url = _push_run(api, repo_id, run_dir, run_dir.name, f"Upload {dest}")
    console.print(f"[green]已上传 → {url}")
    return 0


def _local_runs(config: Config) -> list[Path]:
    """output/ 下的训练输出（跳过 .hf-fetch 等隐藏目录）。"""
    if not config.output_dir.is_dir():
        return []
    return sorted(path for path in config.output_dir.iterdir() if path.is_dir() and not path.name.startswith("."))


def _upload_all_runs(api: HfApi, repo_id: str, runs: list[Path], yes: bool) -> int:
    sizes = {run: _folder_size(run) for run in runs}
    table = Table(show_header=False, box=None, padding=(0, 2))
    for run in runs:
        table.add_row(run.name, human_size(sizes[run]))
    console.print(table)
    if not yes and not confirm(f"上传这 {len(runs)} 个训练输出（共 {human_size(sum(sizes.values()))}）到 "
                               f"{repo_id} 的 {RUNS_DIR}/？没有变化的文件不会重复上传"):
        return 1
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)
    failed = []
    for i, run in enumerate(runs, 1):
        console.print(f"[bold]({i}/{len(runs)}) {run.name}")
        try:
            _push_run(api, repo_id, run, run.name, f"Upload {RUNS_DIR}/{run.name}")
        except (HfHubHTTPError, OSError) as e:
            console.print(f"[red]上传失败：{e}")
            failed.append(run.name)
    if failed:
        console.print(f"[red]{len(failed)} 个上传失败：{', '.join(failed)}。重新运行同一命令即可")
        return 1
    console.print(f"[green]已上传 → {_repo_url(repo_id, RUNS_DIR)}")
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


def upload(names: list[str], yes: bool, config: Config) -> int:
    """依次上传 names 里的训练输出或数据集；names 为空时上传 output/ 下的全部训练输出。"""
    if not names:
        if not (runs := _local_runs(config)):
            console.print(f"[yellow]{config.output_dir} 下没有训练输出")
            return 1
        return _run_hub(lambda: _upload_all_runs(*_connect(config, write=True), runs, yes))
    exit_code = 0
    for name in names:
        exit_code = _upload_one(name, yes, config) or exit_code
    return exit_code


def _upload_one(name: str, yes: bool, config: Config) -> int:
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


# ---- 训练中自动上传 ----

def _snapshot(source: Path, target: Path) -> None:
    """把 source 的当前内容做成 target 下的快照。上传的同时训练还在写 source（追加日志、删旧的训练状态），
    直接上传 source 可能读到写了一半或者已经删掉的文件。"""
    shutil.rmtree(target, ignore_errors=True)
    for directory, _, files in os.walk(source):
        relative = Path(directory).relative_to(source)
        (target / relative).mkdir(parents=True, exist_ok=True)
        for file in files:
            src, dst = Path(directory) / file, target / relative / file
            try:
                if src.stat().st_size >= SNAPSHOT_COPY_LIMIT:
                    try:
                        os.link(src, dst)
                        continue
                    except OSError:
                        pass
                shutil.copy2(src, dst)
            except FileNotFoundError:  # 刚被删掉的旧训练状态
                pass


class AutoUploader:
    """训练时每保存一次（LoRA 和训练状态），就在后台把训练输出目录同步到仓库的 output/<名字>/，不耽误训练。

    同一时间只传一份；上一次还没传完又保存了，就等它传完再传最新的。上传失败只提示，下次保存时会重新同步。
    """

    def __init__(self, config: Config, run_dir: Path) -> None:
        self.api, self.repo_id = _connect(config, write=True)
        self.api.create_repo(self.repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)
        self.run_dir = run_dir
        self.staging = config.output_dir / UPLOAD_STAGING / run_dir.name
        self._lock = threading.Lock()
        self._pending: str | None = None
        self._thread: threading.Thread | None = None
        # huggingface_hub 自己的进度条和日志会打乱训练进度的显示
        disable_progress_bars()
        hub_logging.set_verbosity_error()

    @classmethod
    def connect(cls, config: Config, run_dir: Path) -> "AutoUploader | None":
        """连接失败时打印原因并返回 None。"""
        try:
            return cls(config, run_dir)
        except HubError as e:
            console.print(f"[red]{e}")
        except HfHubHTTPError as e:
            console.print(f"[red]Hugging Face 返回错误：{e}")
        except OSError as e:
            console.print(f"[red]无法连接 Hugging Face：{e}")
        console.print("[dim]不需要自动上传时用 --no-auto-upload")
        return None

    @property
    def url(self) -> str:
        return _repo_url(self.repo_id, f"{RUNS_DIR}/{self.run_dir.name}")

    def request(self, label: str) -> None:
        """在后台上传，label 是这次保存的说明（如「epoch 4」）。"""
        with self._lock:
            self._pending = label
            if self._thread is None:
                self._thread = threading.Thread(target=self._work, name="auto-upload", daemon=True)
                self._thread.start()

    def finish(self, label: str) -> bool:
        """训练结束后调用：等后台的上传结束，再把最终结果同步上去。"""
        with self._lock:
            self._pending = None
            thread = self._thread
        if thread is not None:
            console.print("[dim]等待正在进行的自动上传 ...")
            thread.join()
        with console.status(f"正在上传{label} ..."):
            return self._push(label, final=True)

    def _work(self) -> None:
        while True:
            with self._lock:
                label, self._pending = self._pending, None
                if label is None:
                    self._thread = None
                    return
            self._push(label)

    def _push(self, label: str, final: bool = False) -> bool:
        try:
            _snapshot(self.run_dir, self.staging)
            _push_run(self.api, self.repo_id, self.staging, self.run_dir.name,
                      f"Upload {RUNS_DIR}/{self.run_dir.name} ({label})")
            console.print(f"[dim]已上传{label} → {self.url}")
            return True
        except Exception as e:  # 后台线程里出错不能打断训练
            retry = "" if final else "下次保存后会再试，"
            console.print(f"[yellow]自动上传（{label}）失败：{e}。{retry}也可以运行 llk --upload {self.run_dir.name}")
            return False
        finally:
            shutil.rmtree(self.staging, ignore_errors=True)


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
