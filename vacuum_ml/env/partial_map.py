from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath

from .room import Room
from .dirt_map import DirtMap


class PartialMap:
    """Fog-of-war map observation: (2, MAP_SIZE, MAP_SIZE) float32.

    Channel 0 (geometry): 0=unknown, 0.5=free space, 1.0=wall or obstacle.
    Channel 1 (dirt):     0=unknown/clean, float=known current dirt level.

    The sensor reveals cells within SENSOR_RADIUS world units of the vacuum.
    """

    MAP_SIZE = 84
    SENSOR_RADIUS = 1.5  # world units

    def __init__(self, room: Room):
        self.room = room
        self._geometry = self._rasterize_geometry()   # (84, 84) pre-baked
        self._revealed = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=bool)
        self._dirt_layer = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, x: float, y: float, dirt_map: DirtMap) -> None:
        """Reveal cells within SENSOR_RADIUS of (x, y) and update their dirt values."""
        prow, pcol = self._world_to_pixel(x, y)
        r_px = max(1, int(self.SENSOR_RADIUS / self.room.width * self.MAP_SIZE))

        r0 = max(0, prow - r_px)
        r1 = min(self.MAP_SIZE, prow + r_px + 1)
        c0 = max(0, pcol - r_px)
        c1 = min(self.MAP_SIZE, pcol + r_px + 1)

        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        dist = np.sqrt((rr - prow) ** 2 + (cc - pcol) ** 2)
        in_sensor = dist <= r_px

        self._revealed[r0:r1, c0:c1] |= in_sensor

        # Update dirt for all pixels in sensor zone
        revealed_rows = rr[in_sensor]
        revealed_cols = cc[in_sensor]
        wx = (revealed_cols + 0.5) / self.MAP_SIZE * self.room.width
        wy = (revealed_rows + 0.5) / self.MAP_SIZE * self.room.height
        self._dirt_layer[revealed_rows, revealed_cols] = dirt_map.current_dirt_grid(wx, wy)

    def get_array(self) -> np.ndarray:
        """Returns (2, MAP_SIZE, MAP_SIZE) float32 copy."""
        geom = np.where(self._revealed, self._geometry, 0.0).astype(np.float32)
        dirt = np.where(self._revealed, self._dirt_layer, 0.0).astype(np.float32)
        return np.stack([geom, dirt], axis=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        col = int(np.clip(x / self.room.width * self.MAP_SIZE, 0, self.MAP_SIZE - 1))
        row = int(np.clip(y / self.room.height * self.MAP_SIZE, 0, self.MAP_SIZE - 1))
        return row, col

    def _rasterize_geometry(self) -> np.ndarray:
        """Pre-bake room geometry into (MAP_SIZE, MAP_SIZE). Called once at init."""
        xs = (np.arange(self.MAP_SIZE) + 0.5) / self.MAP_SIZE * self.room.width
        ys = (np.arange(self.MAP_SIZE) + 0.5) / self.MAP_SIZE * self.room.height
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])

        poly_path = MplPath(np.array(self.room.polygon.exterior.coords))
        in_poly = poly_path.contains_points(pts).reshape(self.MAP_SIZE, self.MAP_SIZE)

        in_obs = np.zeros(self.MAP_SIZE * self.MAP_SIZE, dtype=bool)
        for obs in self.room.obstacles:
            obs_path = MplPath(np.array(obs.exterior.coords))
            in_obs |= obs_path.contains_points(pts)
        in_obs = in_obs.reshape(self.MAP_SIZE, self.MAP_SIZE)

        # 0.5 = free space, 1.0 = wall or obstacle (including outside polygon)
        geom = np.ones((self.MAP_SIZE, self.MAP_SIZE), dtype=np.float32)  # default: wall
        geom[in_poly & ~in_obs] = 0.5
        return geom
