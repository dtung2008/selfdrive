"""Build fixed-size observation vectors from world state.

This module converts the raw simulator state (ego vehicle + nearby traffic)
into a fixed-length numeric vector suitable for consumption by a neural-network
policy or a classical planner.  The observation follows an ego-centric frame:
all neighbor positions and speeds are expressed *relative* to the ego vehicle.

Key design choice:
    The neighbor list has a fixed size ``k`` (from config).  When fewer than k
    neighbors exist, the extra slots are zero-filled and marked with
    ``exists_flag = 0`` so the downstream model can distinguish real neighbors
    from padding.
"""
from typing import List
import numpy as np

from utils.types import VehicleState, Observation
from utils.config import ObservationConfig


class ObservationBuilder:
    """Constructs an ``Observation`` from the ego state and a list of other vehicles.

    The builder sorts all visible vehicles by absolute longitudinal distance
    to the ego, picks the closest ``k`` neighbors, and encodes each one as a
    4-tuple ``(rel_x, rel_lane, rel_speed, exists_flag)``.

    Attributes:
        config: An ``ObservationConfig`` holding observation hyper-parameters
                (e.g. ``k_neighbors``).
    """

    def __init__(self, config: ObservationConfig = None):
        """Initialise the builder.

        Args:
            config: Observation configuration.  Falls back to default
                    ``ObservationConfig()`` when *None*.
        """
        self.config = config or ObservationConfig()

    @property
    def k(self) -> int:
        """Number of nearest neighbors to include in the observation."""
        return self.config.k_neighbors

    @property
    def obs_size(self) -> int:
        """Total length of the flat observation vector.

        Returns:
            The number of floats in the vector produced by
            ``Observation.to_vector()``.
        """
        return Observation.vector_size(
            self.k, self.config.ego_features,
            self.config.features_per_neighbor)

    def build(self, ego: VehicleState,
              others: List[VehicleState]) -> Observation:
        """Build an ego-centric observation for the current timestep.

        Neighbors are sorted by absolute longitudinal distance to the ego,
        closest first.  Each neighbor slot is a 4-element row:
            [0] rel_x     — longitudinal offset (positive = ahead)
            [1] rel_lane  — lane difference (positive = to the right)
            [2] rel_speed — speed difference (positive = faster than ego)
            [3] exists    — 1.0 if this slot holds a real vehicle, else 0.0

        Args:
            ego:    The ego vehicle's current state.
            others: States of all other (NPC) vehicles visible in the world.

        Returns:
            An ``Observation`` dataclass that can be flattened to a numpy
            vector via ``to_vector()``.
        """
        # Sort others by distance to ego (closest first)
        sorted_others = sorted(others, key=lambda v: abs(v.x - ego.x))

        # Pre-allocate a (k, fpn) zero matrix; unfilled rows stay as zeros
        # with exists_flag = 0, acting as padding for the neural network.
        fpn = self.config.features_per_neighbor
        neighbors = np.zeros((self.k, fpn), dtype=np.float32)

        for i, v in enumerate(sorted_others[:self.k]):
            neighbors[i, 0] = v.x - ego.x          # relative longitudinal position
            neighbors[i, 1] = v.lane - ego.lane     # relative lane index
            neighbors[i, 2] = v.speed - ego.speed   # relative speed
            neighbors[i, 3] = v.accel - ego.accel   # relative acceleration
            neighbors[i, fpn - 1] = 1.0             # exists flag (always last column)

        return Observation(
            ego_speed=ego.speed,
            ego_lane=ego.lane,
            ego_x=ego.x,
            ego_accel=ego.accel,
            neighbors=neighbors,
        )
