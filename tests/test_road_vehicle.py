"""Tests for environment/road.py and environment/vehicle.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.config import RoadConfig, VehicleConfig
from utils.types import VehicleState, Action, LongitudinalAction, LateralAction
from environment.road import Road
from environment.vehicle import VehicleDynamics


class TestRoad:
    def test_default_two_lanes(self):
        road = Road()
        assert road.num_lanes == 2

    def test_custom_lanes(self):
        road = Road(RoadConfig(num_lanes=4))
        assert road.num_lanes == 4

    def test_lane_center(self):
        road = Road(RoadConfig(num_lanes=3, lane_width=4.0))
        assert abs(road.lane_center_y(0) - 2.0) < 1e-6
        assert abs(road.lane_center_y(1) - 6.0) < 1e-6
        assert abs(road.lane_center_y(2) - 10.0) < 1e-6

    def test_valid_lane(self):
        road = Road(RoadConfig(num_lanes=2))
        assert road.is_valid_lane(0)
        assert road.is_valid_lane(1)
        assert not road.is_valid_lane(-1)
        assert not road.is_valid_lane(2)

    def test_clamp_lane(self):
        road = Road(RoadConfig(num_lanes=3))
        assert road.clamp_lane(-1) == 0
        assert road.clamp_lane(5) == 2
        assert road.clamp_lane(1) == 1

    def test_total_width(self):
        road = Road(RoadConfig(num_lanes=3, lane_width=3.7))
        assert abs(road.total_width - 11.1) < 1e-6


class TestVehicleDynamics:
    def setup_method(self):
        self.dyn = VehicleDynamics(VehicleConfig(
            max_speed=35.0, min_speed=0.0,
            max_acceleration=3.0, max_deceleration=5.0))

    def test_keep_speed(self):
        v = VehicleState(x=100.0, lane=0, speed=20.0)
        a = Action(LongitudinalAction.KEEP, LateralAction.KEEP)
        v2 = self.dyn.step(v, a, dt=0.1, num_lanes=2)
        assert abs(v2.speed - 20.0) < 1e-6
        assert v2.x > v.x  # moved forward
        assert v2.lane == 0

    def test_accelerate(self):
        v = VehicleState(x=100.0, lane=0, speed=20.0)
        a = Action(LongitudinalAction.ACCELERATE, LateralAction.KEEP)
        v2 = self.dyn.step(v, a, dt=0.1, num_lanes=2)
        assert v2.speed > 20.0

    def test_decelerate(self):
        v = VehicleState(x=100.0, lane=0, speed=20.0)
        a = Action(LongitudinalAction.DECELERATE, LateralAction.KEEP)
        v2 = self.dyn.step(v, a, dt=0.1, num_lanes=2)
        assert v2.speed < 20.0

    def test_speed_clamped_min(self):
        v = VehicleState(x=100.0, lane=0, speed=0.5)
        a = Action(LongitudinalAction.DECELERATE, LateralAction.KEEP)
        v2 = self.dyn.step(v, a, dt=1.0, num_lanes=2)
        assert v2.speed >= 0.0

    def test_speed_clamped_max(self):
        v = VehicleState(x=100.0, lane=0, speed=34.0)
        a = Action(LongitudinalAction.ACCELERATE, LateralAction.KEEP)
        v2 = self.dyn.step(v, a, dt=1.0, num_lanes=2)
        assert v2.speed <= 35.0

    def test_lane_change_right(self):
        v = VehicleState(x=100.0, lane=0, speed=20.0)
        a = Action(LongitudinalAction.KEEP, LateralAction.RIGHT)
        v2 = self.dyn.step(v, a, dt=0.1, num_lanes=3)
        assert v2.lane == 1

    def test_lane_change_left(self):
        v = VehicleState(x=100.0, lane=1, speed=20.0)
        a = Action(LongitudinalAction.KEEP, LateralAction.LEFT)
        v2 = self.dyn.step(v, a, dt=0.1, num_lanes=3)
        assert v2.lane == 0

    def test_lane_clamped_left(self):
        v = VehicleState(x=100.0, lane=0, speed=20.0)
        a = Action(LongitudinalAction.KEEP, LateralAction.LEFT)
        v2 = self.dyn.step(v, a, dt=0.1, num_lanes=2)
        assert v2.lane == 0  # can't go left from lane 0

    def test_lane_clamped_right(self):
        v = VehicleState(x=100.0, lane=1, speed=20.0)
        a = Action(LongitudinalAction.KEEP, LateralAction.RIGHT)
        v2 = self.dyn.step(v, a, dt=0.1, num_lanes=2)
        assert v2.lane == 1  # can't go right from lane 1 in 2-lane road

    def test_step_with_acceleration(self):
        v = VehicleState(x=100.0, lane=0, speed=20.0, vehicle_id=5)
        v2 = self.dyn.step_with_acceleration(v, accel=2.0, target_lane=1,
                                              dt=0.1, num_lanes=3)
        assert v2.speed > 20.0
        assert v2.lane == 1
        assert v2.vehicle_id == 5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
