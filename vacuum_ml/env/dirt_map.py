from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath

from .room import Room


class DirtMap:
    """Rasterized coverage tracker. 200×200 grid over the room.

    Call step(x, y) every env step to mark the vacuum's path and get reward
    components. Coverage and current dirt levels are derived from pass counts.
    """

    GRID_SIZE = 200

    def __init__(self, room: Room, vacuum_radius: float = 0.3):
        self.room = room
        self.vacuum_radius = vacuum_radius
        self._pass_count = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)
        self._in_room = self._rasterize_room()
        self._initial_dirt = self._rasterize_dirt()
        weighted = float(self._initial_dirt[self._in_room].sum())
        self._total_weighted_area = weighted if weighted > 1e-6 else 1.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, x: float, y: float) -> tuple[float, float, bool]:
        """Mark vacuum path at (x, y). Returns (first_pass_dirt, second_pass_dirt, is_overvisit).

        Rewards are based on initial dirt at the center point.
        is_overvisit is True when center has been visited 2+ times already.
        """
        row, col = self._world_to_grid(x, y)
        center_passes = int(self._pass_count[row, col])
        initial_dirt = float(self.room.dirt_at(x, y))

        if center_passes == 0:
            first_pass_dirt, second_pass_dirt = initial_dirt, 0.0
        elif center_passes == 1:
            first_pass_dirt, second_pass_dirt = 0.0, initial_dirt * 0.5
        else:
            first_pass_dirt, second_pass_dirt = 0.0, 0.0

        is_overvisit = center_passes >= 2
        self._mark_circle(row, col)
        return first_pass_dirt, second_pass_dirt, is_overvisit

    def mean_coverage(self) -> float:
        """Fraction of dirt-weighted room area cleaned at least once."""
        cleaned = float(self._initial_dirt[(self._in_room) & (self._pass_count >= 1)].sum())
        return cleaned / self._total_weighted_area

    def current_dirt_at(self, x: float, y: float) -> float:
        """Current dirt level at (x, y) after cleaning passes."""
        row, col = self._world_to_grid(x, y)
        passes = int(self._pass_count[row, col])
        initial = float(self._initial_dirt[row, col])
        return float(max(0.0, initial - passes * 0.75))

    def current_dirt_grid(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Vectorized current_dirt_at for arrays of world coordinates."""
        cols = np.clip(
            (xs / self.room.width * self.GRID_SIZE).astype(int), 0, self.GRID_SIZE - 1
        )
        rows = np.clip(
            (ys / self.room.height * self.GRID_SIZE).astype(int), 0, self.GRID_SIZE - 1
        )
        passes = self._pass_count[rows, cols]
        initial = self._initial_dirt[rows, cols]
        return np.maximum(0.0, initial - passes * 0.75)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        col = int(np.clip(x / self.room.width * self.GRID_SIZE, 0, self.GRID_SIZE - 1))
        row = int(np.clip(y / self.room.height * self.GRID_SIZE, 0, self.GRID_SIZE - 1))
        return row, col

    def _mark_circle(self, center_row: int, center_col: int) -> None:
        r = max(1, round(self.vacuum_radius / self.room.width * self.GRID_SIZE))
        r0 = max(0, center_row - r)
        r1 = min(self.GRID_SIZE, center_row + r + 1)
        c0 = max(0, center_col - r)
        c1 = min(self.GRID_SIZE, center_col + r + 1)

        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        dist = np.sqrt((rr - center_row) ** 2 + (cc - center_col) ** 2)
        mask = dist <= r
        self._pass_count[r0:r1, c0:c1] = np.where(
            mask,
            np.minimum(self._pass_count[r0:r1, c0:c1] + 1, 127),
            self._pass_count[r0:r1, c0:c1],
        )

    def _rasterize_room(self) -> np.ndarray:
        xs = (np.arange(self.GRID_SIZE) + 0.5) / self.GRID_SIZE * self.room.width
        ys = (np.arange(self.GRID_SIZE) + 0.5) / self.GRID_SIZE * self.room.height
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])

        poly_path = MplPath(np.array(self.room.polygon.exterior.coords))
        in_poly = poly_path.contains_points(pts).reshape(self.GRID_SIZE, self.GRID_SIZE)

        if self.room.obstacles:
            in_obs = np.zeros(self.GRID_SIZE * self.GRID_SIZE, dtype=bool)
            for obs in self.room.obstacles:
                obs_path = MplPath(np.array(obs.exterior.coords))
                in_obs |= obs_path.contains_points(pts)
            in_obs = in_obs.reshape(self.GRID_SIZE, self.GRID_SIZE)
            return in_poly & ~in_obs

        return in_poly

    def _rasterize_dirt(self) -> np.ndarray:
        xs = (np.arange(self.GRID_SIZE) + 0.5) / self.GRID_SIZE * self.room.width
        ys = (np.arange(self.GRID_SIZE) + 0.5) / self.GRID_SIZE * self.room.height
        xx, yy = np.meshgrid(xs, ys)

        dirt = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        for blob in self.room.blobs:
            dist_sq = (xx - blob.cx) ** 2 + (yy - blob.cy) ** 2
            dirt += blob.peak * np.exp(-dist_sq / (2 * blob.sigma ** 2))

        dirt = np.clip(dirt, 0.0, 1.0).astype(np.float32)
        dirt[~self._in_room] = 0.0
        return dirt
