from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from ...core.config import Config
from ...core.net import request

IMAGE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "webp"})
TAG_CATEGORIES = ("artist", "character", "copyright", "general", "meta")


class BooruError(Exception):
    pass


@dataclass
class Post:
    site: str
    id: int
    md5: str
    file_url: str
    file_ext: str
    width: int
    height: int
    rating: str
    score: int
    tags: dict[str, list[str]] = field(default_factory=dict)
    file_size: int | None = None
    source_url: str = ""
    created_at: str = ""

    @property
    def key(self) -> str:
        return f"{self.site}_{self.id}"

    @property
    def filename(self) -> str:
        return f"{self.key}.{self.file_ext}"

    def caption(self) -> str:
        return ", ".join(tag for category in TAG_CATEGORIES for tag in self.tags.get(category, []))

    def to_record(self) -> dict[str, Any]:
        return {**asdict(self), "file": self.filename}


@dataclass
class Page:
    raw_count: int
    min_id: int | None
    posts: list[Post]


@dataclass
class Sample:
    raw_count: int
    posts: list[Post]


@dataclass
class QueryPlan:
    server: str
    local_include: frozenset[str] = frozenset()
    local_exclude: frozenset[str] = frozenset()


@dataclass
class TagInfo:
    name: str
    category: str
    post_count: int


@dataclass
class Estimate:
    post_count: int
    image_count: int
    total_bytes: int
    exact: bool


def extrapolate(post_count: int, sample: Sample) -> Estimate:
    sizes = [p.file_size for p in sample.posts if p.file_size]
    average = sum(sizes) / len(sizes) if sizes else 0
    post_count = max(post_count, sample.raw_count)
    image_count = round(post_count * len(sample.posts) / sample.raw_count) if sample.raw_count else 0
    return Estimate(post_count, image_count, int(average * image_count), exact=False)


def tally(sample: Sample) -> Estimate:
    sizes = [p.file_size for p in sample.posts if p.file_size]
    average = sum(sizes) / len(sizes) if sizes else 0
    total_bytes = int(sum(sizes) + average * (len(sample.posts) - len(sizes)))
    return Estimate(sample.raw_count, len(sample.posts), total_bytes, exact=len(sizes) == len(sample.posts))


class Booru:
    name: str = ""
    page_size: int = 100
    max_estimate_pages: int = 1

    def __init__(self, client: httpx.AsyncClient, config: Config) -> None:
        self.client = client
        self.config = config
        self._first_pages: list[Page] = []
        self._fully_fetched = False
        self.local_include: frozenset[str] = frozenset()
        self.local_exclude: frozenset[str] = frozenset()

    def keeps(self, post: Post) -> bool:
        tags = {tag for names in post.tags.values() for tag in names}
        return self.local_include <= tags and not self.local_exclude & tags

    async def get_json(self, url: str, params: dict[str, Any]) -> Any:
        response = await request(self.client, "GET", url, params=params)
        if response.status_code >= 400:
            raise BooruError(self.describe_error(response))
        try:
            return response.json()
        except ValueError:
            raise BooruError(f"{self.name} 返回了无法解析的内容: {response.text[:200]}")

    def describe_error(self, response: httpx.Response) -> str:
        try:
            data = response.json()
            message = data.get("message") or data.get("error") or response.text[:200]
        except ValueError:
            message = response.text[:200]
        return f"{self.name} HTTP {response.status_code}: {message}"

    async def count(self, tags: str) -> int:
        raise NotImplementedError

    async def fetch_page(self, tags: str, before_id: int | None) -> Page:
        raise NotImplementedError

    async def random_sample(self, tags: str, pages: int = 2) -> Sample:
        raise NotImplementedError

    async def search_tags(self, query: str, limit: int) -> list[TagInfo]:
        raise NotImplementedError

    async def plan_query(self, tags: str, include: frozenset[str], exclude: frozenset[str]) -> QueryPlan:
        raise NotImplementedError

    async def estimate(self, tags: str) -> Estimate:
        post_count = await self.count(tags)
        self._first_pages.clear()
        self._fully_fetched = False
        fits_in_pages = post_count <= self.page_size * self.max_estimate_pages
        before_id = None
        for _ in range(self.max_estimate_pages if fits_in_pages else 1):
            page = await self.fetch_page(tags, before_id)
            if page.raw_count == 0:
                self._fully_fetched = True
                break
            self._first_pages.append(page)
            before_id = page.min_id

        fetched = Sample(sum(p.raw_count for p in self._first_pages), [post for p in self._first_pages for post in p.posts])
        if self._fully_fetched:
            return tally(fetched)
        try:
            sample = await self.random_sample(tags)
        except BooruError:
            sample = fetched
        return extrapolate(post_count, sample if sample.raw_count else fetched)

    async def quick_estimate(self, tags: str) -> Estimate:
        post_count = await self.count(tags)
        if post_count == 0:
            return Estimate(0, 0, 0, exact=True)
        sample = await self.random_sample(tags, pages=1)
        if 0 < sample.raw_count < self.page_size and sample.raw_count >= post_count:
            return tally(sample)
        return extrapolate(post_count, sample)

    async def iter_posts(self, tags: str, id_below: int | None = None, id_above: int | None = None) -> AsyncIterator[Post]:
        def in_range(post: Post) -> bool:
            return (id_below is None or post.id < id_below) and (id_above is None or post.id > id_above)

        cursor: int | None = None
        for page in self._first_pages:
            for post in page.posts:
                if in_range(post):
                    yield post
            cursor = page.min_id
            if id_above is not None and cursor is not None and cursor <= id_above:
                return
        if self._fully_fetched:
            return
        if id_below is not None and (cursor is None or cursor > id_below):
            cursor = id_below
        while True:
            page = await self.fetch_page(tags, cursor)
            if page.raw_count == 0 or page.min_id is None:
                return
            for post in page.posts:
                if id_above is not None and post.id <= id_above:
                    return
                yield post
            cursor = page.min_id
