"""Tests for environment/observation.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from utils.types import VehicleState, Observation
from utils.config import ObservationConfig
from environment.observation import ObservationBuilder


class TestObservationBuilder:
    def setup_method(self):
        self.builder = ObservationBuilder(ObservationConfig(k_neighbors=4))

    def test_obs_shape_no_neighbors(self):
        ego = VehicleState(x=100, lane=0, speed=20.0)
        obs = self.builder.build(ego, [])
        vec = obs.to_vector()
        ef = self.builder.config.ego_features
        fpn = self.builder.config.features_per_neighbor
        assert vec.shape == (ef + 4 * fpn,)  # ego + 4 neighbors * fpn features

    def test_obs_ego_values(self):
        ego = VehicleState(x=100, lane=1, speed=25.0)
        obs = self.builder.build(ego, [])
        assert abs(obs.ego_speed - 25.0) < 1e-6
        assert obs.ego_lane == 1
        assert abs(obs.ego_x - 100.0) < 1e-6

    def test_obs_neighbors_sorted_by_distance(self):
        ego = VehicleState(x=100, lane=0, speed=20.0, vehicle_id=0)
        v_far = VehicleState(x=200, lane=0, speed=20.0, vehicle_id=1)
        v_close = VehicleState(x=110, lane=1, speed=22.0, vehicle_id=2)
        obs = self.builder.build(ego, [v_far, v_close])
        # First neighbor should be the closer one
        assert abs(obs.neighbors[0, 0] - 10.0) < 1e-6  # rel_x = 110-100
        assert abs(obs.neighbors[1, 0] - 100.0) < 1e-6  # rel_x = 200-100

    def test_obs_padding(self):
        """Extra neighbor slots should be zeros."""
        ego = VehicleState(x=100, lane=0, speed=20.0, vehicle_id=0)
        v1 = VehicleState(x=120, lane=0, speed=20.0, vehicle_id=1)
        obs = self.builder.build(ego, [v1])
        fpn = self.builder.config.features_per_neighbor
        # Slot 0 has data, slots 1-3 should be zero
        assert obs.neighbors[0, fpn - 1] == 1.0  # exists flag
        assert obs.neighbors[1, fpn - 1] == 0.0  # padding
        assert obs.neighbors[2, fpn - 1] == 0.0
        assert obs.neighbors[3, fpn - 1] == 0.0

    def test_relative_values(self):
        ego = VehicleState(x=100, lane=1, speed=20.0, vehicle_id=0)
        v1 = VehicleState(x=130, lane=0, speed=25.0, vehicle_id=1)
        obs = self.builder.build(ego, [v1])
        assert abs(obs.neighbors[0, 0] - 30.0) < 1e-6   # rel_x
        assert abs(obs.neighbors[0, 1] - (-1)) < 1e-6    # rel_lane (0-1)
        assert abs(obs.neighbors[0, 2] - 5.0) < 1e-6     # rel_speed

    def test_truncation_to_k(self):
        """If more vehicles than k, only keep k closest."""
        ego = VehicleState(x=100, lane=0, speed=20.0, vehicle_id=0)
        others = [VehicleState(x=100 + i * 10, lane=0, speed=20.0,
                               vehicle_id=i+1) for i in range(10)]
        obs = self.builder.build(ego, others)
        fpn = self.builder.config.features_per_neighbor
        # Should have exactly k=4 neighbors
        exist_count = sum(obs.neighbors[:, fpn - 1] > 0.5)
        assert exist_count == 4

    def test_obs_size_property(self):
        ef = self.builder.config.ego_features
        fpn = self.builder.config.features_per_neighbor
        assert self.builder.obs_size == ef + 4 * fpn


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
