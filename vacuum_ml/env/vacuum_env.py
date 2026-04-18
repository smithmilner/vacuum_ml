from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .room import Room
from .dirt_map import DirtMap
from .partial_map import PartialMap

MAX_SPEED = 0.5          # world units / step
VACUUM_RADIUS = 0.3      # world units
DOCK_RADIUS = 0.5        # world units — must be within this to dock
DOCK_SPEED_THRESHOLD = 0.05  # forward_speed below this = docking intent
CHARGE_STEPS = 100       # steps to fully recharge
LOW_BATTERY = 0.2        # threshold below which docking charges instead of terminates


class VacuumEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        width: float = 10.0,
        height: float = 10.0,
        max_steps: int = 2000,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0.0, high=1.0, shape=(2, 84, 84), dtype=np.float32),
            "sensors": spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0,  1.0,  1.0], dtype=np.float32),
                dtype=np.float32,
            ),
        })

        # Set by reset()
        self.room: Room
        self.dirt_map: DirtMap
        self.partial_map: PartialMap
        self.x: float
        self.y: float
        self.theta: float
        self.battery: float
        self.steps: int
        self.dock_x: float
        self.dock_y: float
        self._charging: bool
        self._charge_steps_remaining: int
        self._last_bumper_left: float
        self._last_bumper_right: float

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        room_seed = int(self.np_random.integers(0, 2 ** 31))
        self.room = Room(self.width, self.height, seed=room_seed)
        self.dirt_map = DirtMap(self.room, vacuum_radius=VACUUM_RADIUS)
        self.partial_map = PartialMap(self.room)

        self.dock_x = float(self.room.dock.x)
        self.dock_y = float(self.room.dock.y)
        self.x = self.dock_x
        self.y = self.dock_y
        self.theta = 0.0
        self.battery = 1.0
        self.steps = 0
        self._charging = False
        self._charge_steps_remaining = 0
        self._last_bumper_left = 0.0
        self._last_bumper_right = 0.0

        # Mark starting position
        self.dirt_map.step(self.x, self.y)
        self.partial_map.update(self.x, self.y, self.dirt_map)

        return self._obs(), {}

    def step(self, action: np.ndarray):
        if self._charging:
            return self._do_charge_step()

        turn_delta = float(action[0])
        forward_speed = float(action[1])

        self.theta = (self.theta + turn_delta * np.pi) % (2 * np.pi)

        new_x = self.x + np.cos(self.theta) * forward_speed * MAX_SPEED
        new_y = self.y + np.sin(self.theta) * forward_speed * MAX_SPEED

        collided = False
        if self.room.contains(new_x, new_y):
            self.x, self.y = new_x, new_y
            self._last_bumper_left = 0.0
            self._last_bumper_right = 0.0
        else:
            collided = True
            bl, br = self._detect_bumpers()
            self._last_bumper_left = float(bl)
            self._last_bumper_right = float(br)

        # Battery drain
        if forward_speed < 0.05:
            self.battery -= 0.001
        elif forward_speed < 0.2:
            self.battery -= 0.001
        else:
            self.battery -= 0.002
        self.battery = max(0.0, self.battery)

        # Update maps
        first_pass, second_pass, is_overvisit = self.dirt_map.step(self.x, self.y)
        self.partial_map.update(self.x, self.y, self.dirt_map)

        self.steps += 1

        # Reward
        reward = -0.005
        if collided:
            reward -= 0.3
        reward += first_pass
        reward += second_pass
        if is_overvisit and not collided:
            reward -= 0.05

        # Check dock
        dist_dock = np.hypot(self.x - self.dock_x, self.y - self.dock_y)
        at_dock = dist_dock < DOCK_RADIUS and forward_speed < DOCK_SPEED_THRESHOLD

        terminated = False
        truncated = False

        if at_dock:
            if self.battery < LOW_BATTERY:
                reward += 0.5
                self._charging = True
                self._charge_steps_remaining = CHARGE_STEPS
            else:
                coverage = self.dirt_map.mean_coverage()
                reward += 10.0 * coverage
                terminated = True

        if self.battery <= 0.0 and not at_dock:
            reward -= 5.0
            truncated = True

        if self.steps >= self.max_steps:
            truncated = True

        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _do_charge_step(self) -> tuple:
        self._charge_steps_remaining -= 1
        self.battery = min(1.0, self.battery + 1.0 / CHARGE_STEPS)
        self.steps += 1
        if self._charge_steps_remaining <= 0:
            self._charging = False
        truncated = self.steps >= self.max_steps
        return self._obs(), 0.0, False, truncated, self._info()

    def _detect_bumpers(self) -> tuple[bool, bool]:
        left_angle = self.theta + np.pi / 4
        right_angle = self.theta - np.pi / 4
        probe = VACUUM_RADIUS * 0.5
        bl = not self.room.contains(
            self.x + np.cos(left_angle) * probe,
            self.y + np.sin(left_angle) * probe,
        )
        br = not self.room.contains(
            self.x + np.cos(right_angle) * probe,
            self.y + np.sin(right_angle) * probe,
        )
        if not bl and not br:
            bl = br = True
        return bl, br

    def _dock_bearing(self) -> tuple[float, float]:
        dx = self.dock_x - self.x
        dy = self.dock_y - self.y
        bearing = np.arctan2(dy, dx)
        relative = (bearing - self.theta + np.pi) % (2 * np.pi) - np.pi
        return float(np.sin(relative)), float(np.cos(relative))

    def _sensors(self) -> np.ndarray:
        sin_b, cos_b = self._dock_bearing()
        return np.array([
            self._last_bumper_left,
            self._last_bumper_right,
            self.dirt_map.current_dirt_at(self.x, self.y),
            self.battery,
            sin_b,
            cos_b,
        ], dtype=np.float32)

    def _obs(self) -> dict:
        return {
            "map": self.partial_map.get_array(),
            "sensors": self._sensors(),
        }

    def _info(self) -> dict:
        return {
            "coverage": self.dirt_map.mean_coverage(),
            "steps": self.steps,
            "battery": self.battery,
        }
