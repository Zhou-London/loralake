from typing import Any

from .base import IMAGE_EXTENSIONS, TAG_CATEGORIES, Booru, BooruError, Page, Post, QueryPlan, Sample, TagInfo

DEFAULT_TAG_LIMIT = 2
FREE_METATAGS = ("id:", "rating:")
CATEGORY_NAMES = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta"}
RATING_NAMES = {"g": "general", "s": "sensitive", "q": "questionable", "e": "explicit"}
POST_FIELDS = ",".join([
    "id", "md5", "file_url", "file_ext", "file_size", "image_width", "image_height", "rating", "score", "source", "created_at",
    *(f"tag_string_{category}" for category in TAG_CATEGORIES),
])
TAG_FIELDS = "name,post_count,category"


class Danbooru(Booru):
    name = "danbooru"
    rating_names = RATING_NAMES
    base_url = "https://danbooru.donmai.us"
    page_size = 200
    max_estimate_pages = 5

    def _auth_params(self) -> dict[str, str]:
        if self.config.has_danbooru_auth:
            return {"login": self.config.danbooru_login, "api_key": self.config.danbooru_api_key}
        return {}

    async def _get(self, endpoint: str, params: dict[str, Any]) -> Any:
        return await self.get_json(f"{self.base_url}/{endpoint}", {**params, **self._auth_params()})

    async def count(self, tags: str) -> int:
        data = await self._get("counts/posts.json", {"tags": tags})
        counts = data.get("counts")
        if not isinstance(counts, dict) or "posts" not in counts:
            raise BooruError(f"{self.name} 没有返回 \"{tags}\" 的帖子数 (统计超时或查询无效)，请稍后重试")
        return int(counts["posts"] or 0)

    async def tag_query_limit(self) -> int:
        if not self.config.has_danbooru_auth:
            return DEFAULT_TAG_LIMIT
        profile = await self._get("profile.json", {})
        return int(profile.get("tag_query_limit") or DEFAULT_TAG_LIMIT)

    async def plan_query(self, tags: str, include: frozenset[str], exclude: frozenset[str]) -> QueryPlan:
        def counts_against_limit(term: str) -> bool:
            return not term.lstrip("-").startswith(FREE_METATAGS)

        exclude = frozenset(t.lstrip("-") for t in exclude)
        terms = [*tags.split(), *sorted(t for t in include if ":" in t), *sorted(f"-{t}" for t in exclude if ":" in t)]
        include = frozenset(t for t in include if ":" not in t)
        exclude = frozenset(t for t in exclude if ":" not in t)
        slots = await self.tag_query_limit() - sum(map(counts_against_limit, terms))
        post_counts: dict[str, int] = {}
        if slots > 0 and include | exclude:
            items = await self._get("tags.json", {"search[name_comma]": ",".join(include | exclude),
                                                  "only": "name,post_count", "limit": len(include | exclude)})
            post_counts = {item["name"]: item.get("post_count") or 0 for item in items}
        rarest_includes = sorted(include, key=lambda t: post_counts.get(t, 0))
        largest_excludes = sorted(exclude, key=lambda t: -post_counts.get(t, 0))
        server = [*rarest_includes, *(f"-{t}" for t in largest_excludes)][:max(slots, 0)]
        self.local_include = include - set(server)
        self.local_exclude = exclude - {t[1:] for t in server}
        return QueryPlan(" ".join([*terms, *server]), self.local_include, self.local_exclude)

    async def fetch_page(self, tags: str, before_id: int | None) -> Page:
        params: dict[str, Any] = {"tags": tags, "limit": self.page_size, "only": POST_FIELDS}
        if before_id is not None:
            params["page"] = f"b{before_id}"
        items = await self._get("posts.json", params)
        return Page(
            raw_count=len(items),
            min_id=min((item["id"] for item in items), default=None),
            posts=self._parse_posts(items),
        )

    async def random_sample(self, tags: str, pages: int = 2) -> Sample:
        sample = Sample(0, [])
        for _ in range(pages):
            items = await self._get("posts.json", {"tags": f"{tags} order:random", "limit": self.page_size, "only": POST_FIELDS})
            sample.raw_count += len(items)
            sample.posts += self._parse_posts(items)
        return sample

    async def search_tags(self, query: str, limit: int) -> list[TagInfo]:
        found: dict[str, TagInfo] = {}

        def add(name: str | None, category: int | None, post_count: int | None) -> None:
            if name and post_count and name not in found and len(found) < limit:
                found[name] = TagInfo(name, CATEGORY_NAMES.get(category or 0, "general"), post_count)

        def add_tags(items: list[dict[str, Any]]) -> None:
            for item in items:
                add(item.get("name"), item.get("category"), item.get("post_count"))

        suggestions = await self._get("autocomplete.json", {"search[query]": query, "search[type]": "tag_query", "limit": limit})
        for item in suggestions:
            if str(item.get("type", "")).startswith("tag"):
                add(item.get("value"), item.get("category"), item.get("post_count"))

        if len(found) < limit and not query.isascii():
            wiki_pages = await self._get("wiki_pages.json", {"search[other_names_match]": f"*{query}*", "only": "title", "limit": limit})
            titles = [page["title"] for page in wiki_pages if page.get("title") not in found]
            if titles:
                add_tags(await self._get("tags.json", {
                    "search[name_comma]": ",".join(titles), "search[order]": "count",
                    "search[post_count]": ">0", "only": TAG_FIELDS, "limit": limit,
                }))

        fallback_searches = [
            {"search[name_or_alias_matches]": query if "*" in query else f"*{query}*", "search[order]": "count"},
            {"search[fuzzy_name_matches]": query, "search[order]": "similarity"},
        ]
        for search in fallback_searches:
            if len(found) >= limit:
                break
            add_tags(await self._get("tags.json", {**search, "search[post_count]": ">0", "only": TAG_FIELDS, "limit": limit}))
        return list(found.values())

    def _parse_posts(self, items: list[dict[str, Any]]) -> list[Post]:
        return [post for post in map(self._parse_post, items) if post and self.keeps(post)]

    def _parse_post(self, item: dict[str, Any]) -> Post | None:
        file_ext = (item.get("file_ext") or "").lower()
        file_url = item.get("file_url")
        if not file_url or file_ext not in IMAGE_EXTENSIONS:
            return None
        return Post(
            site=self.name,
            id=item["id"],
            md5=item.get("md5") or "",
            file_url=file_url,
            file_ext=file_ext,
            width=item.get("image_width") or 0,
            height=item.get("image_height") or 0,
            rating=RATING_NAMES.get(item.get("rating") or "", ""),
            score=item.get("score") or 0,
            tags={category: (item.get(f"tag_string_{category}") or "").split() for category in TAG_CATEGORIES},
            file_size=item.get("file_size"),
            source_url=item.get("source") or "",
            created_at=item.get("created_at") or "",
        )
