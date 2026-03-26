"""Tests for environment/traffic_manager.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config import TrafficConfig, RoadConfig
from utils.types import VehicleState
from environment.road import Road
from environment.traffic_manager import TrafficManager


class TestTrafficManager:
    def setup_method(self):
        self.road = Road(RoadConfig(num_lanes=2))
        self.tc = TrafficConfig(num_npcs=5, min_spawn_gap=10.0)
        self.tm = TrafficManager(self.road, self.tc, seed=42)

    def test_reset_spawns_npcs(self):
        states = self.tm.reset(ego_x=100.0)
        assert len(states) > 0
        assert len(states) <= self.tc.num_npcs

    def test_npcs_have_valid_lanes(self):
        states = self.tm.reset(ego_x=100.0)
        for s in states:
            assert 0 <= s.lane < self.road.num_lanes

    def test_npcs_have_positive_speed(self):
        states = self.tm.reset(ego_x=100.0)
        for s in states:
            assert s.speed > 0

    def test_step_returns_states(self):
        self.tm.reset(ego_x=100.0)
        ego = VehicleState(x=100.0, lane=0, speed=20.0, vehicle_id=0)
        states = self.tm.step(ego, dt=0.1)
        assert isinstance(states, list)

    def test_step_advances_positions(self):
        self.tm.reset(ego_x=100.0)
        states_before = self.tm.get_states()
        positions_before = {s.vehicle_id: s.x for s in states_before}
        ego = VehicleState(x=100.0, lane=0, speed=20.0, vehicle_id=0)
        self.tm.step(ego, dt=0.1)
        states_after = self.tm.get_states()
        # At least some vehicles should have moved
        moved = 0
        for s in states_after:
            if s.vehicle_id in positions_before:
                if abs(s.x - positions_before[s.vehicle_id]) > 0.01:
                    moved += 1
        assert moved > 0

    def test_despawn_far_vehicles(self):
        """Vehicles far from ego should be removed."""
        self.tm.reset(ego_x=100.0)
        ego = VehicleState(x=100.0, lane=0, speed=20.0, vehicle_id=0)
        # Step many times so ego "moves" far
        for _ in range(100):
            ego = VehicleState(x=ego.x + 50, lane=0, speed=20.0, vehicle_id=0)
            self.tm.step(ego, dt=0.1)
        # Should maintain roughly num_npcs (respawning)
        assert len(self.tm.get_states()) <= self.tc.num_npcs + 2

    def test_no_overlapping_spawns(self):
        """Spawned vehicles should have minimum gap."""
        states = self.tm.reset(ego_x=100.0)
        for i, s1 in enumerate(states):
            for s2 in states[i+1:]:
                if s1.lane == s2.lane:
                    assert abs(s1.x - s2.x) >= self.tc.min_spawn_gap * 0.9

    def test_multi_lane_road(self):
        road = Road(RoadConfig(num_lanes=4))
        tm = TrafficManager(road, TrafficConfig(num_npcs=10), seed=99)
        states = tm.reset(ego_x=100.0)
        lanes_used = set(s.lane for s in states)
        # With 10 NPCs on 4 lanes, should use more than 1 lane
        assert len(lanes_used) > 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
