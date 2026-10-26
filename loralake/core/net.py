import asyncio
import random
import re
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from .. import __version__
from .config import Config

USER_AGENT = f"loralake/{__version__}"
RETRY_STATUS = frozenset({500, 502, 503, 504, 520, 521, 522, 524})
RATE_LIMIT_PAUSE = 60.0
MAX_RATE_LIMIT_RETRIES = 30


class RequestError(Exception):
    pass


def backoff_delay(attempt: int, cap: float = 60.0) -> float:
    return min(cap, 2.0**attempt) + random.random()


class SiteGate:
    def __init__(self) -> None:
        self.paused_until = 0.0
        self.times_paused = 0

    def remaining_pause(self) -> float:
        return max(0.0, self.paused_until - time.monotonic())

    async def wait(self) -> None:
        while (remaining := self.remaining_pause()) > 0:
            await asyncio.sleep(remaining)

    def pause(self, response: httpx.Response | None = None) -> None:
        seconds = max(RATE_LIMIT_PAUSE, _server_requested_pause(response))
        now = time.monotonic()
        if now >= self.paused_until:
            self.times_paused += 1
            self.paused_until = now + seconds


def _server_requested_pause(response: httpx.Response | None) -> float:
    if response is None:
        return 0.0
    retry_after = response.headers.get("Retry-After", "")
    if retry_after.isdigit():
        return float(retry_after)
    reset = re.search(r";t=(\d+)", response.headers.get("RateLimit", ""))
    return float(reset.group(1)) + 1 if reset else 0.0


_GATES: dict[str, SiteGate] = {}


def _site_of(url: str) -> str:
    return ".".join((urlsplit(url).hostname or "").split(".")[-2:])


def gate_for(url: str) -> SiteGate:
    return _GATES.setdefault(_site_of(url), SiteGate())


def paused_sites() -> list[tuple[str, float]]:
    return [(site, gate.remaining_pause()) for site, gate in _GATES.items() if gate.remaining_pause() > 0]


def total_pauses() -> int:
    return sum(gate.times_paused for gate in _GATES.values())


def make_client(config: Config) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(30.0, connect=15.0),
        follow_redirects=True,
        proxy=config.proxy or None,
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
    )


async def request(client: httpx.AsyncClient, method: str, url: str, retries: int = 5, **kwargs: Any) -> httpx.Response:
    gate = gate_for(url)
    last_error: Exception | None = None
    attempt = rate_limited = 0
    while attempt <= retries:
        await gate.wait()
        try:
            response = await client.request(method, url, **kwargs)
        except httpx.TransportError as e:
            last_error = e
            attempt += 1
            await asyncio.sleep(backoff_delay(attempt))
            continue
        if response.status_code == 429 and rate_limited < MAX_RATE_LIMIT_RETRIES:
            rate_limited += 1
            gate.pause(response)
            continue
        if response.status_code in RETRY_STATUS and attempt < retries:
            attempt += 1
            await asyncio.sleep(backoff_delay(attempt))
            continue
        return response
    raise RequestError(f"{method} {url} 失败: {last_error!r}")
