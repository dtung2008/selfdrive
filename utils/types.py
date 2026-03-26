"""Core data types for the self-driving simulator.

This module defines the fundamental data structures used throughout the
self-driving simulation system, including:

- **Action space**: A discrete 3x3 grid combining longitudinal control
  (decelerate / keep / accelerate) with lateral control (lane-change left /
  keep lane / lane-change right), yielding 9 possible actions indexed 0-8.
- **Vehicle state**: Position, lane, speed, and physical dimensions.
- **Observation**: A fixed-size vector encoding ego state plus the K
  nearest neighbour vehicles (relative coordinates), suitable for feeding
  directly into neural-network policies.
- **Transition / Trajectory**: Standard RL bookkeeping for experience
  replay and episode-level statistics.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class LongitudinalAction(IntEnum):
    """Discrete longitudinal (forward/backward) control for the ego vehicle.

    Values are -1, 0, +1 so they can be used directly as signed multipliers
    when computing acceleration deltas.
    """
    DECELERATE = -1  # Apply braking force (up to max_deceleration)
    KEEP = 0         # Maintain current speed
    ACCELERATE = 1   # Apply throttle (up to max_acceleration)


class LateralAction(IntEnum):
    """Discrete lateral (lane-change) control for the ego vehicle.

    Values are -1, 0, +1 where *left* means moving toward lane index 0
    (the leftmost lane) and *right* means moving toward higher lane indices.
    """
    LEFT = -1   # Initiate a lane change to the left (lane index - 1)
    KEEP = 0    # Stay in the current lane
    RIGHT = 1   # Initiate a lane change to the right (lane index + 1)


@dataclass
class Action:
    """Combined longitudinal + lateral ego action.

    The two axes are independent, so the full discrete action space is the
    Cartesian product of {DECELERATE, KEEP, ACCELERATE} x {LEFT, KEEP, RIGHT},
    giving **9** unique actions.

    **Flat-index encoding scheme** (used by ``to_index`` / ``from_index``)::

        lon \\ lat   LEFT(0)  KEEP(1)  RIGHT(2)
        DECEL (0)      0        1        2
        KEEP  (1)      3        4        5
        ACCEL (2)      6        7        8

    The mapping is: ``index = (longitudinal + 1) * 3 + (lateral + 1)``
    where each raw enum value in {-1, 0, 1} is shifted to {0, 1, 2}.

    Attributes:
        longitudinal: Longitudinal control command (default: KEEP).
        lateral: Lateral control command (default: KEEP).
    """
    longitudinal: LongitudinalAction = LongitudinalAction.KEEP
    lateral: LateralAction = LateralAction.KEEP

    def to_index(self) -> int:
        """Convert this action pair to a flat integer index in [0, 8].

        Returns:
            int: The flat index computed as ``(lon + 1) * 3 + (lat + 1)``.
        """
        # Shift enum values from {-1, 0, 1} to {0, 1, 2} so they can be
        # used as row / column indices in the 3x3 action grid.
        lon_idx = self.longitudinal + 1  # 0,1,2
        lat_idx = self.lateral + 1       # 0,1,2
        # Row-major flattening: longitudinal selects the row, lateral the column.
        return lon_idx * 3 + lat_idx

    @staticmethod
    def from_index(idx: int) -> "Action":
        """Reconstruct an Action from a flat index in [0, 8].

        This is the inverse of ``to_index``.

        Args:
            idx: Flat action index (0-8 inclusive).

        Returns:
            Action: The corresponding (longitudinal, lateral) action pair.
        """
        # Reverse the row-major encoding: integer division gives the row
        # (longitudinal), modulo gives the column (lateral).
        lon_idx = idx // 3
        lat_idx = idx % 3
        # Shift back from {0, 1, 2} to the original enum range {-1, 0, 1}.
        return Action(
            longitudinal=LongitudinalAction(lon_idx - 1),
            lateral=LateralAction(lat_idx - 1),
        )

    @staticmethod
    def num_actions() -> int:
        """Return the total number of discrete actions (always 9).

        Returns:
            int: Size of the discrete action space (3 longitudinal x 3 lateral).
        """
        return 9


# ---------------------------------------------------------------------------
# Vehicle state
# ---------------------------------------------------------------------------

@dataclass
class VehicleState:
    """Kinematic state and physical dimensions of a single vehicle.

    The coordinate system uses:
    - **x** (longitudinal): distance along the road in meters, increasing in
      the direction of travel.
    - **lane** (lateral): integer lane index where 0 is the leftmost lane.
    - **speed**: scalar longitudinal speed in m/s (no lateral speed component).

    Attributes:
        x: Longitudinal position along the road (meters).
        lane: Lane index; 0 is the leftmost lane.
        speed: Current longitudinal speed (m/s, non-negative).
        length: Bumper-to-bumper vehicle length (meters). Default 4.5 m
            (~typical sedan).
        width: Side-to-side vehicle width (meters). Default 2.0 m.
        vehicle_id: Unique integer identifier for this vehicle. 0 is
            conventionally used for the ego vehicle.
    """
    x: float           # longitudinal position (meters)
    lane: int           # lane index (0 = leftmost)
    speed: float        # longitudinal speed (m/s)
    length: float = 4.5 # vehicle length (meters)
    width: float = 2.0  # vehicle width (meters)
    vehicle_id: int = 0

    def y_center(self, lane_width: float) -> float:
        """Compute the lateral (y) center of this vehicle.

        The y-axis origin is at the left road edge. Each lane occupies
        ``lane_width`` meters, so the center of lane *i* sits at
        ``(i + 0.5) * lane_width``.

        Args:
            lane_width: Width of a single lane in meters (see
                ``RoadConfig.lane_width``).

        Returns:
            float: Lateral center position in meters from the left road edge.
        """
        return (self.lane + 0.5) * lane_width


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """Fixed-size observation vector suitable for neural-network input.

    The observation is composed of two parts:

    1. **Ego state** (3 floats): ``[ego_speed, ego_lane, ego_x]``
    2. **Neighbour block** (K x 4 floats): for each of the K nearest
       neighbours, a row ``[rel_x, rel_lane, rel_speed, exists]`` where:

       - ``rel_x``: longitudinal offset from ego (positive = ahead).
       - ``rel_lane``: lane difference from ego (positive = to the right).
       - ``rel_speed``: speed difference (neighbour speed minus ego speed).
       - ``exists``: 1.0 if this slot holds a real vehicle, 0.0 if it is
         a padding slot (fewer than K neighbours visible). The other three
         fields are zeroed when ``exists == 0``.

    When flattened via ``to_vector()``, the total length is ``3 + K * 4``.

    Attributes:
        ego_speed: Ego vehicle's current speed (m/s).
        ego_lane: Ego vehicle's current lane index.
        ego_x: Ego vehicle's longitudinal position (meters).
        neighbors: Numpy array of shape ``(K, 4)`` holding the neighbour
            feature rows described above.
    """
    ego_speed: float
    ego_lane: int
    ego_x: float
    # For each of K neighbours: (rel_x, rel_lane, rel_speed, exists)
    neighbors: np.ndarray  # shape (K, 4)

    def to_vector(self) -> np.ndarray:
        """Flatten the observation into a contiguous 1-D numpy array.

        The layout is::

            [ego_speed, ego_lane, ego_x,
             n0_rel_x, n0_rel_lane, n0_rel_speed, n0_exists,
             n1_rel_x, n1_rel_lane, n1_rel_speed, n1_exists,
             ...
             nK_rel_x, nK_rel_lane, nK_rel_speed, nK_exists]

        Returns:
            np.ndarray: 1-D float array of length ``3 + K * 4``.
        """
        # Pack the three ego scalars into a small array, then concatenate
        # with the flattened neighbour matrix for a single contiguous vector.
        ego = np.array([self.ego_speed, float(self.ego_lane), self.ego_x])
        return np.concatenate([ego, self.neighbors.flatten()])

    @staticmethod
    def vector_size(k_neighbors: int) -> int:
        """Compute the observation vector length for a given neighbour count.

        Args:
            k_neighbors: Number of neighbour slots (K).

        Returns:
            int: Total vector length (``3 + k_neighbors * 4``).
        """
        return 3 + k_neighbors * 4


# ---------------------------------------------------------------------------
# RL bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """A single (s, a, r, s', done) environment transition.

    Used as the fundamental unit for experience-replay buffers and batch
    training of policies and world models.

    Attributes:
        obs: Flattened observation vector at time *t* (from
            ``Observation.to_vector()``).
        action_idx: Flat discrete action index in [0, 8] (see
            ``Action.to_index()``).
        reward: Scalar reward received after taking the action.
        next_obs: Flattened observation vector at time *t + 1*.
        done: True if the episode ended after this transition (collision or
            max steps reached).
    """
    obs: np.ndarray
    action_idx: int
    reward: float
    next_obs: np.ndarray
    done: bool


@dataclass
class Trajectory:
    """A complete episode represented as an ordered list of transitions.

    Provides convenience methods for appending transitions during rollout
    and querying episode-level statistics.

    Attributes:
        transitions: Ordered list of ``Transition`` objects from the first
            step to the terminal step of the episode.
    """
    transitions: List[Transition] = field(default_factory=list)

    def add(self, obs, action_idx, reward, next_obs, done):
        """Append a new transition to the trajectory.

        Args:
            obs: Flattened observation vector at time *t*.
            action_idx: Flat discrete action index in [0, 8].
            reward: Scalar reward received.
            next_obs: Flattened observation vector at time *t + 1*.
            done: Whether the episode terminated after this step.
        """
        self.transitions.append(Transition(obs, action_idx, reward, next_obs, done))

    def __len__(self):
        """Return the number of transitions (i.e. timesteps) in the episode."""
        return len(self.transitions)

    @property
    def total_reward(self) -> float:
        """Compute the undiscounted cumulative reward for the episode.

        Returns:
            float: Sum of all transition rewards.
        """
        return sum(t.reward for t in self.transitions)
