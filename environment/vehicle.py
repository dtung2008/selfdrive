"""Vehicle state and dynamics (kinematic model)."""
from dataclasses import dataclass
from utils.config import VehicleConfig
from utils.types import VehicleState, Action, LongitudinalAction, LateralAction


class VehicleDynamics:
    """Applies actions to vehicle states using simple kinematics."""

    def __init__(self, config: VehicleConfig = None):
        self.config = config or VehicleConfig()

    def step(self, state: VehicleState, action: Action, dt: float,
             num_lanes: int) -> VehicleState:
        """Advance vehicle state by one timestep.

        Args:
            state: Current vehicle state.
            action: Action to apply.
            dt: Timestep duration in seconds.
            num_lanes: Total number of lanes on the road.

        Returns:
            New VehicleState after applying action.
        """
        # Longitudinal dynamics
        accel = self._get_acceleration(action.longitudinal)
        new_speed = state.speed + accel * dt
        new_speed = max(self.config.min_speed,
                        min(new_speed, self.config.max_speed))
        new_x = state.x + new_speed * dt

        # Lateral dynamics (discrete lane change)
        new_lane = state.lane
        if action.lateral == LateralAction.LEFT:
            new_lane = max(0, state.lane - 1)
        elif action.lateral == LateralAction.RIGHT:
            new_lane = min(num_lanes - 1, state.lane + 1)

        return VehicleState(
            x=new_x,
            lane=new_lane,
            speed=new_speed,
            length=state.length,
            width=state.width,
            vehicle_id=state.vehicle_id,
        )

    def step_with_acceleration(self, state: VehicleState, accel: float,
                                target_lane: int, dt: float,
                                num_lanes: int) -> VehicleState:
        """Step with raw acceleration value (used by IDM/MOBIL).

        Args:
            state: Current vehicle state.
            accel: Acceleration in m/s^2 (can be negative).
            target_lane: Lane to move to.
            dt: Timestep.
            num_lanes: Total lanes.

        Returns:
            New VehicleState.
        """
        new_speed = state.speed + accel * dt
        new_speed = max(self.config.min_speed,
                        min(new_speed, self.config.max_speed))
        new_x = state.x + new_speed * dt
        new_lane = max(0, min(target_lane, num_lanes - 1))

        return VehicleState(
            x=new_x,
            lane=new_lane,
            speed=new_speed,
            length=state.length,
            width=state.width,
            vehicle_id=state.vehicle_id,
        )

    def _get_acceleration(self, lon_action: LongitudinalAction) -> float:
        """Map discrete longitudinal action to acceleration value."""
        if lon_action == LongitudinalAction.ACCELERATE:
            return self.config.max_acceleration
        elif lon_action == LongitudinalAction.DECELERATE:
            return -self.config.max_deceleration
        else:
            return 0.0
