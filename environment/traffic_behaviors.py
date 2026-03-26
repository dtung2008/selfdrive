"""Traffic behaviour models for NPC vehicles.

Hierarchy:
  TrafficBehavior (abstract)
  ├── ConstantSpeedBehavior   — maintain fixed speed
  ├── IDMBehavior             — Intelligent Driver Model (car-following)
  └── MOBILBehavior           — MOBIL lane change + IDM longitudinal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

from utils.types import VehicleState


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _find_leader(vehicle: VehicleState,
                 others: List[VehicleState],
                 lane: int) -> Optional[VehicleState]:
    """Find the nearest vehicle ahead in the given lane."""
    candidates = [v for v in others
                  if v.lane == lane and v.x > vehicle.x
                  and v.vehicle_id != vehicle.vehicle_id]
    if not candidates:
        return None
    return min(candidates, key=lambda v: v.x)


def _find_follower(vehicle: VehicleState,
                   others: List[VehicleState],
                   lane: int) -> Optional[VehicleState]:
    """Find the nearest vehicle behind in the given lane."""
    candidates = [v for v in others
                  if v.lane == lane and v.x < vehicle.x
                  and v.vehicle_id != vehicle.vehicle_id]
    if not candidates:
        return None
    return max(candidates, key=lambda v: v.x)


def _gap(front: VehicleState, behind: VehicleState) -> float:
    """Net bumper-to-bumper gap (meters)."""
    return front.x - behind.x - front.length


# ---------------------------------------------------------------------------
# abstract base
# ---------------------------------------------------------------------------

class TrafficBehavior(ABC):
    """Interface for NPC driving behaviours.

    Returns (acceleration, target_lane).
    """

    @abstractmethod
    def get_action(self, vehicle: VehicleState,
                   all_vehicles: List[VehicleState],
                   num_lanes: int) -> Tuple[float, int]:
        """Compute acceleration (m/s^2) and target lane index."""
        ...


# ---------------------------------------------------------------------------
# constant speed
# ---------------------------------------------------------------------------

class ConstantSpeedBehavior(TrafficBehavior):
    """Maintain a fixed speed, stay in lane."""

    def __init__(self, target_speed: float = 25.0):
        self.target_speed = target_speed

    def get_action(self, vehicle, all_vehicles, num_lanes):
        # gentle P-controller towards target speed
        accel = 1.0 * (self.target_speed - vehicle.speed)
        accel = max(-3.0, min(accel, 2.0))
        return accel, vehicle.lane


# ---------------------------------------------------------------------------
# IDM (Intelligent Driver Model)
# ---------------------------------------------------------------------------

@dataclass
class IDMParams:
    desired_speed: float = 28.0   # v0
    time_headway: float = 1.5     # T
    min_gap: float = 2.0          # s0
    max_accel: float = 2.0        # a
    comfortable_decel: float = 3.0 # b
    delta: float = 4.0            # acceleration exponent

    def with_personality(self, rng) -> "IDMParams":
        """Return a copy with slight random personality variation."""
        import copy
        p = copy.copy(self)
        std = 0.15
        p.desired_speed *= max(0.7, rng.normal(1.0, std))
        p.time_headway *= max(0.5, rng.normal(1.0, std))
        p.max_accel *= max(0.5, rng.normal(1.0, std))
        p.comfortable_decel *= max(0.5, rng.normal(1.0, std))
        return p


class IDMBehavior(TrafficBehavior):
    """Intelligent Driver Model — car-following only, no lane changes."""

    def __init__(self, params: IDMParams = None):
        self.p = params or IDMParams()

    def compute_accel(self, vehicle: VehicleState,
                      leader: Optional[VehicleState]) -> float:
        """IDM acceleration calculation."""
        v = vehicle.speed
        v0 = self.p.desired_speed
        a = self.p.max_accel
        delta = self.p.delta

        # free-road term
        accel = a * (1.0 - (v / max(v0, 1e-6)) ** delta)

        if leader is not None:
            s = _gap(leader, vehicle)
            s = max(s, 0.1)  # avoid division by zero
            dv = v - leader.speed
            s_star = (self.p.min_gap
                      + v * self.p.time_headway
                      + v * dv / (2.0 * math.sqrt(a * self.p.comfortable_decel)))
            s_star = max(s_star, self.p.min_gap)
            accel -= a * (s_star / s) ** 2

        return accel

    def get_action(self, vehicle, all_vehicles, num_lanes):
        leader = _find_leader(vehicle, all_vehicles, vehicle.lane)
        accel = self.compute_accel(vehicle, leader)
        # clamp
        accel = max(-self.p.comfortable_decel * 1.5, min(accel, self.p.max_accel))
        return accel, vehicle.lane


# ---------------------------------------------------------------------------
# MOBIL (lane change decision model)
# ---------------------------------------------------------------------------

@dataclass
class MOBILParams:
    politeness: float = 0.5       # p — how much we care about others
    threshold: float = 0.2        # Δa_th — min acceleration gain to change
    safe_decel: float = 4.0       # b_safe — max imposed braking on new follower


class MOBILBehavior(TrafficBehavior):
    """MOBIL lane-change model on top of IDM longitudinal control."""

    def __init__(self, idm_params: IDMParams = None,
                 mobil_params: MOBILParams = None):
        self.idm = IDMBehavior(idm_params or IDMParams())
        self.mobil = mobil_params or MOBILParams()

    def get_action(self, vehicle, all_vehicles, num_lanes):
        # IDM accel in current lane
        leader_cur = _find_leader(vehicle, all_vehicles, vehicle.lane)
        accel_cur = self.idm.compute_accel(vehicle, leader_cur)

        best_lane = vehicle.lane
        best_accel = accel_cur

        for target_lane in [vehicle.lane - 1, vehicle.lane + 1]:
            if target_lane < 0 or target_lane >= num_lanes:
                continue

            # What would our accel be in the target lane?
            leader_new = _find_leader(vehicle, all_vehicles, target_lane)
            accel_new = self.idm.compute_accel(vehicle, leader_new)

            # Safety: new follower's deceleration must be acceptable
            follower_new = _find_follower(vehicle, all_vehicles, target_lane)
            if follower_new is not None:
                # Follower's accel if we move in front of them
                accel_follower_after = self.idm.compute_accel(follower_new, vehicle)
                if accel_follower_after < -self.mobil.safe_decel:
                    continue  # unsafe — would force hard braking

                # Follower's current accel
                follower_leader_cur = _find_leader(follower_new, all_vehicles,
                                                   target_lane)
                accel_follower_before = self.idm.compute_accel(
                    follower_new, follower_leader_cur)
            else:
                accel_follower_before = 0.0
                accel_follower_after = 0.0

            # Current follower in our lane — they benefit if we leave
            follower_old = _find_follower(vehicle, all_vehicles, vehicle.lane)
            if follower_old is not None:
                old_follower_leader = _find_leader(follower_old, all_vehicles,
                                                   vehicle.lane)
                accel_old_before = self.idm.compute_accel(
                    follower_old, vehicle)
                accel_old_after = self.idm.compute_accel(
                    follower_old, old_follower_leader)
            else:
                accel_old_before = 0.0
                accel_old_after = 0.0

            # MOBIL incentive criterion
            gain = (accel_new - accel_cur
                    + self.mobil.politeness * (
                        (accel_follower_after - accel_follower_before)
                        + (accel_old_after - accel_old_before)))

            if gain > self.mobil.threshold and accel_new > best_accel:
                best_lane = target_lane
                best_accel = accel_new

        # Use IDM accel for the chosen lane
        if best_lane != vehicle.lane:
            leader_target = _find_leader(vehicle, all_vehicles, best_lane)
            final_accel = self.idm.compute_accel(vehicle, leader_target)
        else:
            final_accel = accel_cur

        final_accel = max(-self.idm.p.comfortable_decel * 1.5,
                          min(final_accel, self.idm.p.max_accel))
        return final_accel, best_lane
