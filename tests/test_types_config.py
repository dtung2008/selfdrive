"""Tests for utils/types.py and utils/config.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from utils.types import (Action, LongitudinalAction, LateralAction,
                          VehicleState, Observation, Transition, Trajectory)
from utils.config import Config, DEFAULT_CONFIG


class TestAction:
    def test_action_index_roundtrip(self):
        """All 9 actions should survive to_index / from_index roundtrip."""
        for idx in range(9):
            a = Action.from_index(idx)
            assert a.to_index() == idx, f"Roundtrip failed for index {idx}"

    def test_action_values(self):
        a = Action(LongitudinalAction.ACCELERATE, LateralAction.LEFT)
        assert a.longitudinal == LongitudinalAction.ACCELERATE
        assert a.lateral == LateralAction.LEFT

    def test_num_actions(self):
        assert Action.num_actions() == 9

    def test_default_action(self):
        a = Action()
        assert a.longitudinal == LongitudinalAction.KEEP
        assert a.lateral == LateralAction.KEEP
        assert a.to_index() == 4  # center of 3x3 grid


class TestVehicleState:
    def test_y_center(self):
        v = VehicleState(x=100.0, lane=1, speed=20.0)
        y = v.y_center(lane_width=3.7)
        assert abs(y - 1.5 * 3.7) < 1e-6

    def test_defaults(self):
        v = VehicleState(x=0, lane=0, speed=0)
        assert v.length == 4.5
        assert v.width == 2.0


class TestObservation:
    def test_to_vector_shape(self):
        k = 6
        obs = Observation(ego_speed=20.0, ego_lane=1, ego_x=100.0,
                          neighbors=np.zeros((k, 4)))
        vec = obs.to_vector()
        assert vec.shape == (Observation.vector_size(k),)
        assert len(vec) == 3 + k * 4

    def test_vector_size(self):
        assert Observation.vector_size(6) == 27
        assert Observation.vector_size(0) == 3


class TestTrajectory:
    def test_add_and_length(self):
        traj = Trajectory()
        assert len(traj) == 0
        traj.add(np.zeros(5), 0, 1.0, np.zeros(5), False)
        traj.add(np.zeros(5), 1, 2.0, np.zeros(5), True)
        assert len(traj) == 2
        assert abs(traj.total_reward - 3.0) < 1e-6


class TestConfig:
    def test_default_config(self):
        c = DEFAULT_CONFIG
        assert c.road.num_lanes == 2
        assert c.sim.dt > 0
        assert c.obs.k_neighbors > 0

    def test_custom_config(self):
        c = Config()
        c.road.num_lanes = 4
        assert c.road.num_lanes == 4


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
