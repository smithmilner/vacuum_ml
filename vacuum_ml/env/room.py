from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


@dataclass
class GaussianBlob:
    cx: float
    cy: float
    sigma: float
    peak: float


class Room:
    """Shapely-polygon room with Gaussian dirt blobs.

    Coordinates: x = horizontal (0..width), y = vertical (0..height).
    Dock is always at (1.0, 1.0) — near bottom-left, always inside polygon.
    """

    def __init__(
        self,
        width: float = 10.0,
        height: float = 10.0,
        obstacle_count: int = 3,
        blob_count: int = 5,
        seed: int | None = None,
    ):
        self.width = width
        self.height = height
        rng = np.random.default_rng(seed)

        self.polygon: Polygon = self._make_polygon(rng)
        self.dock: Point = Point(1.0, 1.0)
        self.obstacles: list[Polygon] = self._make_obstacles(rng, obstacle_count)
        self.blobs: list[GaussianBlob] = self._make_blobs(rng, blob_count)

        obs_union = unary_union(self.obstacles) if self.obstacles else Polygon()
        self._interior: Polygon = self.polygon.difference(obs_union)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def contains(self, x: float, y: float) -> bool:
        """True if (x, y) is inside room and not inside any obstacle."""
        pt = Point(x, y)
        if not self.polygon.contains(pt):
            return False
        return not any(obs.contains(pt) for obs in self.obstacles)

    def dirt_at(self, x: float, y: float) -> float:
        """Initial dirt level at (x, y) from Gaussian blobs, clamped to [0, 1]."""
        total = sum(
            b.peak * np.exp(-((x - b.cx) ** 2 + (y - b.cy) ** 2) / (2 * b.sigma ** 2))
            for b in self.blobs
        )
        return float(np.clip(total, 0.0, 1.0))

    @property
    def cleanable_area(self) -> float:
        return float(self._interior.area)

    # ------------------------------------------------------------------
    # Private construction helpers
    # ------------------------------------------------------------------

    def _make_polygon(self, rng: np.random.Generator) -> Polygon:
        corners = [
            (0.0, 0.0),
            (self.width, 0.0),
            (self.width, self.height),
            (0.0, self.height),
        ]
        if rng.random() < 0.2:
            idx = int(rng.integers(1, 4))  # never perturb corner 0 (dock area)
            dx = rng.uniform(-self.width * 0.15, self.width * 0.15)
            dy = rng.uniform(-self.height * 0.15, self.height * 0.15)
            corners[idx] = (corners[idx][0] + dx, corners[idx][1] + dy)
        poly = Polygon(corners)
        if not poly.is_valid:
            poly = Polygon([(0, 0), (self.width, 0), (self.width, self.height), (0, self.height)])
        return poly

    def _make_obstacles(self, rng: np.random.Generator, count: int) -> list[Polygon]:
        obstacles: list[Polygon] = []
        inner = self.polygon.buffer(-1.5)
        dock_exclusion = self.dock.buffer(2.0)
        if inner.is_empty:
            return obstacles
        for _ in range(count):
            for _ in range(20):
                cx = rng.uniform(2.0, self.width - 2.0)
                cy = rng.uniform(2.0, self.height - 2.0)
                w = rng.uniform(0.5, 1.5)
                h = rng.uniform(0.5, 1.5)
                obs = Polygon([
                    (cx - w / 2, cy - h / 2),
                    (cx + w / 2, cy - h / 2),
                    (cx + w / 2, cy + h / 2),
                    (cx - w / 2, cy + h / 2),
                ])
                if not inner.contains(obs):
                    continue
                if dock_exclusion.intersects(obs):
                    continue
                existing = unary_union(obstacles) if obstacles else Polygon()
                if obs.intersects(existing):
                    continue
                obstacles.append(obs)
                break
        return obstacles

    def _make_blobs(self, rng: np.random.Generator, count: int) -> list[GaussianBlob]:
        return [
            GaussianBlob(
                cx=rng.uniform(0.5, self.width - 0.5),
                cy=rng.uniform(0.5, self.height - 0.5),
                sigma=rng.uniform(1.0, 3.0),
                peak=rng.uniform(0.3, 0.7),
            )
            for _ in range(count)
        ]
