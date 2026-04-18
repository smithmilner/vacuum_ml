import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .room import Room

# (row_delta, col_delta) for actions: 0=up, 1=down, 2=left, 3=right
_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class VacuumEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        max_steps: int = 200,
        obstacle_density: float = 0.1,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.obstacle_density = obstacle_density
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        # 3 channels: obstacles, cleanliness, vacuum position
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, height, width), dtype=np.float32
        )

        # Set by reset()
        self.room: Room
        self.pos: tuple[int, int]
        self.steps: int
        self.cleaned: np.ndarray

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        room_seed = int(self.np_random.integers(0, 2**31))
        self.room = Room(self.width, self.height, self.obstacle_density, seed=room_seed)
        self.pos = (0, 0)
        self.steps = 0
        self.cleaned = np.zeros((self.height, self.width), dtype=bool)
        self._clean_current()
        return self._obs(), {}

    def step(self, action: int):
        dr, dc = _ACTIONS[int(action)]
        r, c = self.pos
        nr, nc = r + dr, c + dc

        reward = -0.01  # time penalty per step

        if (
            0 <= nr < self.height
            and 0 <= nc < self.width
            and not self.room.obstacles[nr, nc]
        ):
            self.pos = (nr, nc)
            if not self.cleaned[nr, nc]:
                dirt = float(self.room.cleanliness[nr, nc])
                reward += 1.0 + dirt  # bonus scales with dirtiness
                self._clean_current()
            else:
                reward -= 0.1  # revisit penalty
        else:
            reward -= 0.5  # wall / obstacle collision

        self.steps += 1
        coverage = float(self.cleaned.sum()) / self.room.cleanable_cells
        terminated = coverage >= 1.0
        truncated = self.steps >= self.max_steps

        return self._obs(), float(reward), terminated, truncated, {"coverage": coverage, "steps": self.steps}

    def _clean_current(self):
        r, c = self.pos
        self.cleaned[r, c] = True
        self.room.cleanliness[r, c] = 0.0

    def _obs(self) -> np.ndarray:
        r, c = self.pos
        room_state = self.room.get_state()  # (H, W, 2)
        pos_channel = np.zeros((self.height, self.width), dtype=np.float32)
        pos_channel[r, c] = 1.0
        # Stack to (3, H, W): obstacles, cleanliness, vacuum position
        return np.stack(
            [room_state[:, :, 0], room_state[:, :, 1], pos_channel],
            axis=0,
        )
