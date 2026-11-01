from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MAX_BUCKETS = 9
MAX_CANDIDATES = 256


@dataclass(frozen=True)
class BucketPlan:
    aspect_ratios: tuple[float, ...]
    images: tuple[int, ...]
    batch_size: int
    dropped: int
    mean_crop: float

    @property
    def steps_per_epoch(self) -> int:
        return sum(count // self.batch_size for count in self.images)


def image_aspect_ratio(path: Path) -> float:
    with Image.open(path) as image:
        width, height = image.size
    return width / height


class _Search:
    # 代价 = 丢弃的图片数 + 裁掉的面积（以整张图为单位），分桶规则与 diffusion-pipe 一致：
    # 每张图进入 log 空间中最近的桶（平局取较小的宽高比），每个桶只保留 batch 大小整数倍的图片。

    def __init__(self, aspect_ratios: Iterable[float], batch_size: int):
        counter = Counter(aspect_ratios)
        self.ratios = np.array(sorted(counter))
        self.log_ratios = np.log(self.ratios)
        self.counts = np.array([counter[ratio] for ratio in self.ratios])
        self.batch_size = batch_size
        cumulative = np.cumsum(self.counts) / self.counts.sum()
        self.candidates = sorted({int(np.searchsorted(cumulative, q))
                                  for q in np.linspace(0, 1, MAX_CANDIDATES, endpoint=False)})
        if len(self.ratios) <= MAX_CANDIDATES:
            self.candidates = list(range(len(self.ratios)))

    def evaluate(self, centers: tuple[int, ...]) -> tuple[float, np.ndarray, int, float]:
        log_centers = self.log_ratios[list(centers)]
        distance = np.abs(self.log_ratios[:, None] - log_centers[None, :])
        nearest = np.argmin(distance, axis=1)
        images = np.bincount(nearest, weights=self.counts, minlength=len(centers)).astype(int)
        dropped = int((images % self.batch_size).sum())
        crop = float((self.counts * (1 - np.exp(-distance[np.arange(len(nearest)), nearest]))).sum())
        return dropped + crop, images, dropped, crop

    def initial(self, buckets: int) -> tuple[int, ...]:
        cumulative = np.cumsum(self.counts) / self.counts.sum()
        quantiles = (np.arange(buckets) + 0.5) / buckets
        return tuple(sorted({int(np.searchsorted(cumulative, q)) for q in quantiles}))

    def refine(self, centers: tuple[int, ...]) -> tuple[int, ...]:
        best_cost = self.evaluate(centers)[0]
        improved = True
        while improved:
            improved = False
            for position in range(len(centers)):
                for candidate in self.candidates:
                    if candidate in centers:
                        continue
                    moved = tuple(sorted(centers[:position] + (candidate,) + centers[position + 1:]))
                    if (cost := self.evaluate(moved)[0]) < best_cost - 1e-9:
                        centers, best_cost, improved = moved, cost, True
        return centers


def plan_buckets(aspect_ratios: Iterable[float], batch_size: int, max_buckets: int = MAX_BUCKETS) -> BucketPlan:
    search = _Search(aspect_ratios, batch_size)
    candidates = [search.refine(search.initial(buckets))
                  for buckets in range(1, min(max_buckets, len(search.ratios)) + 1)]
    best = min(candidates, key=lambda centers: search.evaluate(centers)[0])
    _, images, dropped, crop = search.evaluate(best)
    used = images > 0
    return BucketPlan(
        aspect_ratios=tuple(float(search.ratios[index]) for index, keep in zip(best, used) if keep),
        images=tuple(int(count) for count in images[used]),
        batch_size=batch_size,
        dropped=dropped,
        mean_crop=crop / int(search.counts.sum()),
    )
