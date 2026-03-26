"""Vehicle state and dynamics (kinematic bicycle-less model).

This module provides the ``VehicleDynamics`` class which applies actions
to a ``VehicleState`` using a simple first-order kinematic model:

    new_speed = old_speed + accel * dt          (clamped to [min, max])
    new_x     = old_x     + new_speed * dt      (Euler integration)

Lateral movement is *discrete*: the vehicle occupies exactly one lane at
a time, and a lane change moves it to an adjacent lane index in a single
timestep.  There is no continuous lateral position or steering angle.

Two stepping interfaces are provided:
    - ``step()``  — takes a discrete ``Action`` (used by the ego vehicle).
    - ``step_with_acceleration()`` — takes a raw float acceleration and
      target lane (used by NPC behaviours like IDM/MOBIL).
"""
from dataclasses import dataclass
from utils.config import VehicleConfig
from utils.types import VehicleState, Action, LongitudinalAction, LateralAction


class VehicleDynamics:
    """Applies actions to vehicle states using simple first-order kinematics.

    The dynamics model is intentionally minimal: no tire slip, no
    steering angle, no inertial lag.  This keeps the simulation fast
    and the action space tractable for RL agents while still capturing
    the essential longitudinal and lateral decisions.

    Attributes:
        config: A ``VehicleConfig`` holding acceleration limits, speed
                bounds, and vehicle dimensions.
    """

    def __init__(self, config: VehicleConfig = None):
        """Initialise with vehicle configuration.

        Args:
            config: Vehicle dynamics configuration.  Defaults to
                    ``VehicleConfig()`` when *None*.
        """
        self.config = config or VehicleConfig()

    def step(self, state: VehicleState, action: Action, dt: float,
             num_lanes: int) -> VehicleState:
        """Advance the vehicle state by one timestep using a discrete action.

        This is the primary interface for the **ego vehicle**, which
        receives actions from the RL agent or planner.

        Longitudinal update:
            accel    = map(action.longitudinal)   (see ``_get_acceleration``)
            speed'   = clamp(speed + accel * dt,  [min_speed, max_speed])
            x'       = x + speed' * dt

        Lateral update (discrete):
            LEFT  -> lane' = max(0, lane - 1)
            RIGHT -> lane' = min(num_lanes-1, lane + 1)
            KEEP  -> lane' = lane

        Args:
            state:     Current vehicle state.
            action:    Discrete ``Action`` with longitudinal and lateral
                       components.
            dt:        Timestep duration in seconds.
            num_lanes: Total number of lanes (for clamping).

        Returns:
            A new ``VehicleState`` reflecting the updated position, lane,
            and speed.  The vehicle's dimensions and ID are preserved.
        """
        # --- Longitudinal dynamics ---
        accel = self._get_acceleration(action.longitudinal)
        new_speed = state.speed + accel * dt
        # Clamp speed to configured bounds (no reversing, no exceeding max)
        new_speed = max(self.config.min_speed,
                        min(new_speed, self.config.max_speed))
        # Euler integration for position
        new_x = state.x + new_speed * dt

        # --- Lateral dynamics (discrete lane change) ---
        new_lane = state.lane
        if action.lateral == LateralAction.LEFT:
            new_lane = max(0, state.lane - 1)           # clamp at leftmost lane
        elif action.lateral == LateralAction.RIGHT:
            new_lane = min(num_lanes - 1, state.lane + 1)  # clamp at rightmost lane

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
        """Advance the vehicle state using a raw acceleration value.

        This is the interface for **NPC vehicles** whose behaviours
        (IDM, MOBIL) produce continuous acceleration values and explicit
        target lane indices rather than discrete action enums.

        The kinematic update is identical to ``step()`` except that
        acceleration is provided directly instead of being mapped from
        a discrete enum.

        Args:
            state:       Current vehicle state.
            accel:       Acceleration in m/s^2 (negative = braking).
            target_lane: Desired lane index after this timestep.
            dt:          Timestep duration in seconds.
            num_lanes:   Total number of lanes (for clamping).

        Returns:
            A new ``VehicleState`` with updated position, lane, and speed.
        """
        # Longitudinal: Euler integration with speed clamping
        new_speed = state.speed + accel * dt
        new_speed = max(self.config.min_speed,
                        min(new_speed, self.config.max_speed))
        new_x = state.x + new_speed * dt

        # Lateral: clamp target lane to valid range
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
        """Map a discrete longitudinal action enum to an acceleration value.

        Args:
            lon_action: One of ``ACCELERATE``, ``DECELERATE``, or
                        ``MAINTAIN``.

        Returns:
            Acceleration in m/s^2:
                - ACCELERATE -> +max_acceleration
                - DECELERATE -> -max_deceleration
                - MAINTAIN   -> 0.0
        """
        if lon_action == LongitudinalAction.ACCELERATE:
            return self.config.max_acceleration
        elif lon_action == LongitudinalAction.DECELERATE:
            return -self.config.max_deceleration
        else:
            return 0.0
