"""Tests for traffic behaviour models: Constant, IDM, MOBIL."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.types import VehicleState
from environment.traffic_behaviors import (
    ConstantSpeedBehavior, IDMBehavior, IDMParams,
    MOBILBehavior, MOBILParams, _find_leader, _find_follower, _gap,
)


# ---------- helpers ----------

class TestHelpers:
    def test_find_leader(self):
        ego = VehicleState(x=100, lane=0, speed=20, vehicle_id=0)
        v1 = VehicleState(x=120, lane=0, speed=20, vehicle_id=1)
        v2 = VehicleState(x=80, lane=0, speed=20, vehicle_id=2)
        v3 = VehicleState(x=150, lane=0, speed=20, vehicle_id=3)
        leader = _find_leader(ego, [v1, v2, v3], lane=0)
        assert leader.vehicle_id == 1  # closest ahead

    def test_find_leader_different_lane(self):
        ego = VehicleState(x=100, lane=0, speed=20, vehicle_id=0)
        v1 = VehicleState(x=110, lane=1, speed=20, vehicle_id=1)
        leader = _find_leader(ego, [v1], lane=0)
        assert leader is None  # v1 is in different lane

    def test_find_follower(self):
        ego = VehicleState(x=100, lane=0, speed=20, vehicle_id=0)
        v1 = VehicleState(x=80, lane=0, speed=20, vehicle_id=1)
        v2 = VehicleState(x=90, lane=0, speed=20, vehicle_id=2)
        follower = _find_follower(ego, [v1, v2], lane=0)
        assert follower.vehicle_id == 2  # closest behind

    def test_gap(self):
        front = VehicleState(x=120, lane=0, speed=20, length=4.5, vehicle_id=1)
        behind = VehicleState(x=100, lane=0, speed=20, length=4.5, vehicle_id=2)
        g = _gap(front, behind)
        assert abs(g - 15.5) < 1e-6  # 20 - 4.5


# ---------- Constant Speed ----------

class TestConstantSpeed:
    def test_stays_in_lane(self):
        behavior = ConstantSpeedBehavior(target_speed=25.0)
        v = VehicleState(x=100, lane=1, speed=25.0, vehicle_id=1)
        accel, lane = behavior.get_action(v, [], num_lanes=3)
        assert lane == 1

    def test_accelerates_when_slow(self):
        behavior = ConstantSpeedBehavior(target_speed=25.0)
        v = VehicleState(x=100, lane=0, speed=20.0, vehicle_id=1)
        accel, lane = behavior.get_action(v, [], num_lanes=2)
        assert accel > 0  # should speed up

    def test_decelerates_when_fast(self):
        behavior = ConstantSpeedBehavior(target_speed=25.0)
        v = VehicleState(x=100, lane=0, speed=30.0, vehicle_id=1)
        accel, lane = behavior.get_action(v, [], num_lanes=2)
        assert accel < 0  # should slow down


# ---------- IDM ----------

class TestIDM:
    def test_free_road_accelerates(self):
        """On empty road, IDM should accelerate toward desired speed."""
        idm = IDMBehavior(IDMParams(desired_speed=28.0))
        v = VehicleState(x=100, lane=0, speed=15.0, vehicle_id=1)
        accel, lane = idm.get_action(v, [], num_lanes=2)
        assert accel > 0

    def test_at_desired_speed_low_accel(self):
        idm = IDMBehavior(IDMParams(desired_speed=28.0))
        v = VehicleState(x=100, lane=0, speed=28.0, vehicle_id=1)
        accel, lane = idm.get_action(v, [], num_lanes=2)
        assert abs(accel) < 0.5  # near zero at desired speed

    def test_close_leader_decelerates(self):
        idm = IDMBehavior(IDMParams(desired_speed=28.0, min_gap=2.0))
        v = VehicleState(x=100, lane=0, speed=25.0, vehicle_id=1)
        leader = VehicleState(x=108, lane=0, speed=15.0, vehicle_id=2)
        accel, lane = idm.get_action(v, [leader], num_lanes=2)
        assert accel < 0  # should brake

    def test_stays_in_lane(self):
        idm = IDMBehavior()
        v = VehicleState(x=100, lane=1, speed=20.0, vehicle_id=1)
        _, lane = idm.get_action(v, [], num_lanes=3)
        assert lane == 1

    def test_personality_variation(self):
        """Personality should produce different params."""
        import numpy as np
        rng = np.random.RandomState(42)
        p1 = IDMParams().with_personality(rng)
        p2 = IDMParams()
        # At least one parameter should differ
        assert (p1.desired_speed != p2.desired_speed or
                p1.time_headway != p2.time_headway)


# ---------- MOBIL ----------

class TestMOBIL:
    def test_no_change_on_empty_road(self):
        mobil = MOBILBehavior()
        v = VehicleState(x=100, lane=0, speed=25.0, vehicle_id=1)
        accel, lane = mobil.get_action(v, [], num_lanes=2)
        assert lane == 0  # no reason to change

    def test_changes_lane_to_pass(self):
        """Should change lane when blocked by slow leader."""
        mobil = MOBILBehavior(
            idm_params=IDMParams(desired_speed=28.0),
            mobil_params=MOBILParams(politeness=0.3, threshold=0.1),
        )
        v = VehicleState(x=100, lane=0, speed=25.0, vehicle_id=1)
        # Slow leader very close in lane 0
        slow = VehicleState(x=112, lane=0, speed=10.0, vehicle_id=2)
        accel, lane = mobil.get_action(v, [slow], num_lanes=2)
        assert lane == 1  # should move to lane 1

    def test_no_change_when_unsafe(self):
        """Don't change if it would force hard braking on follower."""
        mobil = MOBILBehavior(
            mobil_params=MOBILParams(safe_decel=2.0),  # strict safety
        )
        v = VehicleState(x=100, lane=0, speed=25.0, vehicle_id=1)
        slow = VehicleState(x=112, lane=0, speed=10.0, vehicle_id=2)
        # Fast follower very close in target lane
        follower = VehicleState(x=96, lane=1, speed=30.0, vehicle_id=3)
        accel, lane = mobil.get_action(v, [slow, follower], num_lanes=2)
        # With strict safety, should stay put
        assert lane == 0

    def test_respects_lane_bounds(self):
        mobil = MOBILBehavior()
        v = VehicleState(x=100, lane=0, speed=25.0, vehicle_id=1)
        accel, lane = mobil.get_action(v, [], num_lanes=1)
        assert lane == 0  # only one lane, can't change


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
