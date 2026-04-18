import numpy as np


class Room:
    def __init__(
        self,
        width: int,
        height: int,
        obstacle_density: float = 0.1,
        seed: int | None = None,
    ):
        self.width = width
        self.height = height
        rng = np.random.default_rng(seed)

        self.obstacles: np.ndarray = rng.random((height, width)) < obstacle_density
        self.obstacles[0, 0] = False  # start position always clear

        self.cleanliness: np.ndarray = rng.uniform(0.0, 1.0, (height, width)).astype(np.float32)
        self.cleanliness[self.obstacles] = 0.0  # obstacles aren't dirty

    @property
    def cleanable_cells(self) -> int:
        return int((~self.obstacles).sum())

    def get_state(self) -> np.ndarray:
        """Returns (H, W, 2): channel 0 = obstacles, channel 1 = cleanliness."""
        return np.stack(
            [self.obstacles.astype(np.float32), self.cleanliness],
            axis=-1,
        )
