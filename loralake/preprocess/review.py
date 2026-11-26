"""llk --review：在浏览器里整理数据集本身（与模型无关，先于预处理）。

浏览全部图片、批量丢弃、校对修改 tag、确认疑似重复。结果保存在数据集目录的 review.json（见 curation.py），
预处理和训练会先应用它。
"""

import errno
import json
import mimetypes
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from PIL import Image

from ..core.config import Config
from ..core.console import console
from ..core.dataset import METADATA_FILE, Record, available_records, resolve_dataset
from .curation import REVIEW_FILE, Curation, normalize_tags
from .dedup import DedupResult, deduplicate

DEFAULT_PORT = 8765
PORT_ATTEMPTS = 10
HOST = "127.0.0.1"
PAGE_FILE = Path(__file__).with_name("review.html")
THUMB_DIR = ".thumbs"
THUMB_SIZE = 512
# 比较疑似重复时显示在图片下方、有助于判断该留哪张的 tag
INFO_TAGS = (
    "textless_version", "english_text", "japanese_text", "chinese_text", "korean_text", "speech_bubble", "logo",
    "watermark", "sample_watermark", "signature", "artist_name", "copyright_notice", "dated", "cropped",
    "variant_set", "revision", "md5_mismatch", "downscaled", "resized", "jpeg_artifacts",
)
POST_URLS = {"danbooru": "https://danbooru.donmai.us/posts/{id}"}


class RequestError(ValueError):
    pass


def _make_thumbnail(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.draft("RGB", (THUMB_SIZE * 2, THUMB_SIZE * 2))
        image.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
        has_alpha = image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info)
        image = image.convert("RGBA" if has_alpha else "RGB")
        tmp = target.with_name(f"{target.name}.{threading.get_ident()}.tmp")
        image.save(tmp, "WEBP", quality=82, method=4)
    os.replace(tmp, target)


class ReviewSession:
    def __init__(self, dataset_dir: Path, records: list[Record], dedup: DedupResult, curation: Curation,
                 thumb_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.thumb_dir = thumb_dir
        self.records = {record.key: record for record in records if record.key not in dedup.auto_dropped}
        self.groups = dedup.groups
        self.curation = curation
        self.lock = threading.Lock()

    def _image(self, record: Record) -> dict[str, Any]:
        tags = normalize_tags(self.curation.apply(record).tags)
        return {
            "key": record.key, "id": record.id, "url": POST_URLS.get(record.site, "").format(id=record.id),
            "w": record.width, "h": record.height, "rating": record.rating,
            "kb": round((self.dataset_dir / record.file).stat().st_size / 1024),
            "tags": tags, "edited": record.key in self.curation.tags, "dropped": record.key in self.curation.dropped,
        }

    def page_data(self) -> dict[str, Any]:
        records = sorted(self.records.values(), key=lambda record: -record.id)
        return {
            "dataset": self.dataset_dir.name,
            "images": [self._image(record) for record in records],
            "groups": [{"keys": group.keys, "confirmed": self.curation.confirmed(group.pairs)}
                       for group in self.groups],
            "infoTags": INFO_TAGS,
        }

    def image_path(self, key: str) -> Path | None:
        record = self.records.get(key)
        return self.dataset_dir / record.file if record else None

    def thumbnail_path(self, key: str) -> Path | None:
        source = self.image_path(key)
        if source is None or not source.is_file():
            return None
        target = self.thumb_dir / f"{key}.webp"
        if not target.exists() or target.stat().st_mtime_ns < source.stat().st_mtime_ns:
            try:
                _make_thumbnail(source, target)
            except (OSError, ValueError):
                return source
        return target

    def _keys(self, value: Any) -> list[str]:
        if not isinstance(value, list) or not all(isinstance(key, str) and key in self.records for key in value):
            raise RequestError("keys")
        return value

    def _save(self) -> None:
        self.curation.save(self.dataset_dir)

    def drop(self, body: dict[str, Any]) -> dict[str, Any]:
        keys, dropped = self._keys(body.get("keys")), bool(body.get("dropped"))
        with self.lock:
            self.curation.dropped = (self.curation.dropped | set(keys)) if dropped else (self.curation.dropped - set(keys))
            self._save()
        return {}

    def set_tags(self, body: dict[str, Any]) -> dict[str, Any]:
        edits = body.get("tags") or {}
        revert = self._keys(body.get("revert") or [])
        if not isinstance(edits, dict):
            raise RequestError("tags")
        self._keys(list(edits))
        for tags in edits.values():
            if not (isinstance(tags, dict) and all(isinstance(category, str) and isinstance(values, list)
                                                   and all(isinstance(tag, str) for tag in values)
                                                   for category, values in tags.items())):
                raise RequestError("tags")
        with self.lock:
            for key, tags in edits.items():
                self.curation.set_tags(self.records[key], tags)
            for key in revert:
                self.curation.tags.pop(key, None)
            self._save()
            changed = [*edits, *revert]
            return {"images": {key: {"tags": normalize_tags(self.curation.apply(self.records[key]).tags),
                                     "edited": key in self.curation.tags} for key in changed}}

    def confirm(self, body: dict[str, Any]) -> dict[str, Any]:
        index = body.get("group")
        if not isinstance(index, int) or not 0 <= index < len(self.groups):
            raise RequestError("group")
        with self.lock:
            self.curation.reviewed |= set(self.groups[index].pairs)
            self._save()
        return {}


def _handler(session: ReviewSession) -> type[BaseHTTPRequestHandler]:
    actions = {"/api/drop": session.drop, "/api/tags": session.set_tags, "/api/confirm": session.confirm}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send(self, status: int, body: bytes, content_type: str, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: dict[str, Any]) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _file(self, file: Path | None) -> None:
            if file is None or not file.is_file():
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            self._send(HTTPStatus.OK, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "image/webp", True)

        def do_GET(self) -> None:
            path = unquote(self.path.split("?", 1)[0])
            if path == "/":
                data = json.dumps(session.page_data(), ensure_ascii=False).replace("</", "<\\/")
                page = PAGE_FILE.read_text(encoding="utf-8").replace("__DATA__", data)
                self._send(HTTPStatus.OK, page.encode(), "text/html; charset=utf-8")
            elif path.startswith("/img/"):
                self._file(session.image_path(path[5:]))
            elif path.startswith("/thumb/"):
                self._file(session.thumbnail_path(path[7:]))
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self) -> None:
            if (action := actions.get(self.path)) is None:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                if not isinstance(body, dict):
                    raise RequestError("body")
                result = action(body)
            except (ValueError, TypeError) as e:
                self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})
                return
            except OSError as e:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(e)})
                return
            self._json(HTTPStatus.OK, {"ok": True, **result})

    return Handler


def _bind(port: int, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    for candidate in range(port, port + PORT_ATTEMPTS):
        try:
            return ThreadingHTTPServer((HOST, candidate), handler)
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
    raise OSError(errno.EADDRINUSE, f"端口 {port}-{port + PORT_ATTEMPTS - 1} 都被占用，请用 --port 指定其他端口")


def review(dataset: str, port: int, config: Config) -> int:
    dataset_dir = resolve_dataset(dataset, config.dataset_dir)
    if dataset_dir is None:
        console.print(f"[red]找不到数据集 {dataset}（需要包含 {METADATA_FILE}）")
        return 1
    records = available_records(dataset_dir)
    if not records:
        console.print("[yellow]没有可用的图片")
        return 1
    curation = Curation.load(dataset_dir)
    with console.status("正在查重 ..."):
        dedup = deduplicate(dataset_dir, [curation.apply(record) for record in records], curation.reviewed,
                            curation.dropped)
    session = ReviewSession(dataset_dir, records, dedup, curation, config.derived_dir / dataset_dir.name / THUMB_DIR)
    try:
        server = _bind(port, _handler(session))
    except OSError as e:
        console.print(f"[red]无法启动审核页面: {e}")
        return 1
    url = f"http://localhost:{server.server_address[1]}/"
    console.print(f"数据集: {dataset_dir}（{len(session.records):,} 张，像素相同的 {len(dedup.auto_dropped):,} 张已自动去除）")
    console.print(f"疑似重复 {len(dedup.groups):,} 组，其中 {len(dedup.pending):,} 组待确认")
    console.print(f"在浏览器中打开 [bold]{url}[/]")
    console.print(f"[dim]修改实时保存到 {dataset_dir / REVIEW_FILE}，完成后按 Ctrl+C 退出")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    kept = sum(key not in curation.dropped for key in session.records)
    pending = sum(not curation.confirmed(group.pairs) and sum(key not in curation.dropped for key in group.keys) > 1
                  for group in session.groups)
    console.print(f"\n保留 {kept:,} 张，丢弃 {len(session.records) - kept:,} 张，改过 tag 的 {len(curation.tags):,} 张，"
                  f"疑似重复还有 {pending:,} 组待确认")
    return 0
