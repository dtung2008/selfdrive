"""Core data types for the self-driving simulator."""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional
import numpy as np


class LongitudinalAction(IntEnum):
    """Longitudinal control: brake / keep / accelerate."""
    DECELERATE = -1
    KEEP = 0
    ACCELERATE = 1


class LateralAction(IntEnum):
    """Lateral control: lane change left / keep / lane change right."""
    LEFT = -1
    KEEP = 0
    RIGHT = 1


@dataclass
class Action:
    """Combined ego action."""
    longitudinal: LongitudinalAction = LongitudinalAction.KEEP
    lateral: LateralAction = LateralAction.KEEP

    def to_index(self) -> int:
        """Convert to flat index in [0, 8]. Layout: lon in {-1,0,1} x lat in {-1,0,1}."""
        lon_idx = self.longitudinal + 1  # 0,1,2
        lat_idx = self.lateral + 1       # 0,1,2
        return lon_idx * 3 + lat_idx

    @staticmethod
    def from_index(idx: int) -> "Action":
        """Create Action from flat index in [0, 8]."""
        lon_idx = idx // 3
        lat_idx = idx % 3
        return Action(
            longitudinal=LongitudinalAction(lon_idx - 1),
            lateral=LateralAction(lat_idx - 1),
        )

    @staticmethod
    def num_actions() -> int:
        return 9


@dataclass
class VehicleState:
    """State of a single vehicle."""
    x: float           # longitudinal position (meters)
    lane: int           # lane index (0 = leftmost)
    speed: float        # longitudinal speed (m/s)
    length: float = 4.5 # vehicle length (meters)
    width: float = 2.0  # vehicle width (meters)
    vehicle_id: int = 0

    def y_center(self, lane_width: float) -> float:
        """Lateral center position given lane width."""
        return (self.lane + 0.5) * lane_width


@dataclass
class Observation:
    """Fixed-size observation vector for agents.

    Contains ego state followed by relative states of K nearest neighbours.
    """
    ego_speed: float
    ego_lane: int
    ego_x: float
    # For each of K neighbours: (rel_x, rel_lane, rel_speed, exists)
    neighbors: np.ndarray  # shape (K, 4)

    def to_vector(self) -> np.ndarray:
        """Flatten to 1-D numpy array."""
        ego = np.array([self.ego_speed, float(self.ego_lane), self.ego_x])
        return np.concatenate([ego, self.neighbors.flatten()])

    @staticmethod
    def vector_size(k_neighbors: int) -> int:
        return 3 + k_neighbors * 4


@dataclass
class Transition:
    """Single environment transition for replay / training."""
    obs: np.ndarray
    action_idx: int
    reward: float
    next_obs: np.ndarray
    done: bool


@dataclass
class Trajectory:
    """A full episode."""
    transitions: List[Transition] = field(default_factory=list)

    def add(self, obs, action_idx, reward, next_obs, done):
        self.transitions.append(Transition(obs, action_idx, reward, next_obs, done))

    def __len__(self):
        return len(self.transitions)

    @property
    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)
