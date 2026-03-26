"""Build fixed-size observation vectors from world state."""
from typing import List
import numpy as np

from utils.types import VehicleState, Observation
from utils.config import ObservationConfig


class ObservationBuilder:
    """Constructs Observation from ego state and list of other vehicles."""

    def __init__(self, config: ObservationConfig = None):
        self.config = config or ObservationConfig()

    @property
    def k(self) -> int:
        return self.config.k_neighbors

    @property
    def obs_size(self) -> int:
        return Observation.vector_size(self.k)

    def build(self, ego: VehicleState,
              others: List[VehicleState]) -> Observation:
        """Build observation for the ego vehicle.

        Neighbors are sorted by absolute distance, closest first.
        Each neighbor slot: (rel_x, rel_lane, rel_speed, exists_flag).
        """
        # Sort others by distance to ego
        sorted_others = sorted(others, key=lambda v: abs(v.x - ego.x))
        neighbors = np.zeros((self.k, 4), dtype=np.float32)

        for i, v in enumerate(sorted_others[:self.k]):
            neighbors[i, 0] = v.x - ego.x          # relative x
            neighbors[i, 1] = v.lane - ego.lane     # relative lane
            neighbors[i, 2] = v.speed - ego.speed   # relative speed
            neighbors[i, 3] = 1.0                   # exists

        return Observation(
            ego_speed=ego.speed,
            ego_lane=ego.lane,
            ego_x=ego.x,
            neighbors=neighbors,
        )
