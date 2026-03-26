"""Traffic behaviour models for NPC vehicles.

This module implements the driving behaviours assigned to non-player-
controlled (NPC) vehicles.  Each behaviour produces an acceleration and a
target lane at every timestep, which the ``VehicleDynamics`` model then
uses to update the vehicle's kinematic state.

Hierarchy:
    TrafficBehavior (abstract base)
    +-- ConstantSpeedBehavior   -- simple P-controller toward a fixed speed
    +-- IDMBehavior             -- Intelligent Driver Model (car-following)
    +-- MOBILBehavior           -- MOBIL lane-change model + IDM longitudinal

References:
    * Treiber, M., Hennecke, A. & Helbing, D. (2000).
      "Congested traffic states in empirical observations and microscopic
      simulations."  Physical Review E, 62(2), 1805.
      (Original IDM paper.)
    * Kesting, A., Treiber, M. & Helbing, D. (2007).
      "General lane-changing model MOBIL for car-following models."
      Transportation Research Record, 1999(1), 86-94.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

from utils.types import VehicleState


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _find_leader(vehicle: VehicleState,
                 others: List[VehicleState],
                 lane: int) -> Optional[VehicleState]:
    """Find the nearest vehicle ahead of *vehicle* in the given lane.

    "Ahead" means a strictly greater longitudinal position (``x``).
    The vehicle itself is excluded via ``vehicle_id`` comparison.

    Args:
        vehicle: The reference vehicle.
        others:  All vehicles to search through (may include *vehicle*).
        lane:    The lane to search in.

    Returns:
        The ``VehicleState`` of the closest leader, or *None* if the lane
        ahead is clear.
    """
    candidates = [v for v in others
                  if v.lane == lane and v.x > vehicle.x
                  and v.vehicle_id != vehicle.vehicle_id]
    if not candidates:
        return None
    return min(candidates, key=lambda v: v.x)


def _find_follower(vehicle: VehicleState,
                   others: List[VehicleState],
                   lane: int) -> Optional[VehicleState]:
    """Find the nearest vehicle behind *vehicle* in the given lane.

    "Behind" means a strictly smaller longitudinal position (``x``).

    Args:
        vehicle: The reference vehicle.
        others:  All vehicles to search through.
        lane:    The lane to search in.

    Returns:
        The ``VehicleState`` of the closest follower, or *None* if there
        is no trailing vehicle.
    """
    candidates = [v for v in others
                  if v.lane == lane and v.x < vehicle.x
                  and v.vehicle_id != vehicle.vehicle_id]
    if not candidates:
        return None
    return max(candidates, key=lambda v: v.x)


def _gap(front: VehicleState, behind: VehicleState) -> float:
    """Compute the net bumper-to-bumper gap between two vehicles.

    The gap is measured from the rear bumper of *front* to the front
    bumper of *behind*.  This approximation uses the full vehicle length
    rather than separate front/rear overhangs.

    Args:
        front:  The vehicle that is ahead.
        behind: The vehicle that is behind.

    Returns:
        Gap in meters.  Can be negative if the vehicles overlap.
    """
    return front.x - behind.x - front.length


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class TrafficBehavior(ABC):
    """Abstract interface for NPC driving behaviours.

    Every concrete behaviour must implement ``get_action`` which, given the
    current vehicle state and all surrounding traffic, returns an
    ``(acceleration, target_lane)`` tuple.
    """

    @abstractmethod
    def get_action(self, vehicle: VehicleState,
                   all_vehicles: List[VehicleState],
                   num_lanes: int) -> Tuple[float, int]:
        """Compute the desired acceleration and target lane.

        Args:
            vehicle:      The NPC vehicle being controlled.
            all_vehicles: All vehicles in the scene (NPCs + ego).
            num_lanes:    Number of lanes on the road.

        Returns:
            A tuple ``(accel, target_lane)`` where *accel* is in m/s^2
            (negative for braking) and *target_lane* is a lane index.
        """
        ...


# ---------------------------------------------------------------------------
# Constant-speed behaviour
# ---------------------------------------------------------------------------

class ConstantSpeedBehavior(TrafficBehavior):
    """Maintain a fixed target speed using a proportional controller.

    This is the simplest behaviour: no car-following, no lane changes.
    A gentle P-controller adjusts acceleration to converge on
    ``target_speed``, clamped to [-3, +2] m/s^2 for realism.

    Attributes:
        target_speed: Desired cruising speed in m/s.
    """

    def __init__(self, target_speed: float = 25.0):
        """Initialise with the desired cruising speed.

        Args:
            target_speed: Speed to maintain, in m/s.
        """
        self.target_speed = target_speed

    def get_action(self, vehicle, all_vehicles, num_lanes):
        """Return acceleration toward target speed; keep current lane.

        Args:
            vehicle:      Current vehicle state.
            all_vehicles: (unused) All vehicles in the scene.
            num_lanes:    (unused) Number of lanes.

        Returns:
            ``(accel, current_lane)`` — acceleration clamped to [-3, +2].
        """
        # Proportional controller: gain = 1.0
        accel = 1.0 * (self.target_speed - vehicle.speed)
        # Clamp to physically reasonable bounds
        accel = max(-3.0, min(accel, 2.0))
        return accel, vehicle.lane


# ---------------------------------------------------------------------------
# IDM (Intelligent Driver Model)
# ---------------------------------------------------------------------------

@dataclass
class IDMParams:
    """Parameters for the Intelligent Driver Model.

    Symbol mapping (matching the original Treiber et al. 2000 paper):
        v0    = desired_speed        — free-flow desired speed [m/s]
        T     = time_headway         — safe time headway [s]
        s0    = min_gap              — minimum bumper-to-bumper gap [m]
        a     = max_accel            — maximum acceleration [m/s^2]
        b     = comfortable_decel    — comfortable deceleration [m/s^2]
        delta = delta                — acceleration exponent (unitless)
    """
    desired_speed: float = 28.0        # v0 — free-flow desired speed [m/s]
    time_headway: float = 1.5          # T  — desired time headway [s]
    min_gap: float = 2.0               # s0 — minimum spacing at standstill [m]
    max_accel: float = 2.0             # a  — maximum acceleration [m/s^2]
    comfortable_decel: float = 3.0     # b  — comfortable braking decel [m/s^2]
    delta: float = 4.0                 # acceleration exponent (controls how
                                       #     sharply accel drops near v0)

    def with_personality(self, rng) -> "IDMParams":
        """Return a new ``IDMParams`` with randomised personality variation.

        Each parameter is scaled by a Gaussian multiplier (mean 1, std 0.15)
        with a lower clamp to prevent degenerate values (e.g. negative
        desired speed).

        Args:
            rng: A ``numpy.random.RandomState`` for reproducibility.

        Returns:
            A new ``IDMParams`` instance with perturbed values.
        """
        import copy
        p = copy.copy(self)
        std = 0.15
        p.desired_speed *= max(0.7, rng.normal(1.0, std))
        p.time_headway *= max(0.5, rng.normal(1.0, std))
        p.max_accel *= max(0.5, rng.normal(1.0, std))
        p.comfortable_decel *= max(0.5, rng.normal(1.0, std))
        return p


class IDMBehavior(TrafficBehavior):
    """Intelligent Driver Model — car-following only, no lane changes.

    IDM computes a smooth, continuous acceleration that:
      - Accelerates on a free road toward the desired speed ``v0``.
      - Decelerates smoothly when approaching a slower leader.
      - Maintains a speed-dependent safe following distance.

    The model is deterministic given the same leader state, making it
    well-suited for microscopic traffic simulation.

    Attributes:
        p: ``IDMParams`` controlling the model's calibration.
    """

    def __init__(self, params: IDMParams = None):
        """Initialise with IDM calibration parameters.

        Args:
            params: IDM parameters.  Defaults to ``IDMParams()`` when *None*.
        """
        self.p = params or IDMParams()

    def compute_accel(self, vehicle: VehicleState,
                      leader: Optional[VehicleState]) -> float:
        """Compute the IDM acceleration for a vehicle given its leader.

        The IDM acceleration formula is:

            a_IDM = a * [1 - (v/v0)^delta - (s*(v, dv) / s)^2]

        where the *desired dynamic gap* ``s*`` is:

            s*(v, dv) = s0 + v*T + v*dv / (2*sqrt(a*b))

        The first term ``1 - (v/v0)^delta`` is the **free-road** component
        that drives acceleration to zero as speed approaches ``v0``.
        The second term ``(s*/s)^2`` is the **interaction** component that
        causes braking when the gap to the leader shrinks below ``s*``.

        Args:
            vehicle: The vehicle whose acceleration is being computed.
            leader:  The vehicle directly ahead (or *None* for free road).

        Returns:
            Acceleration in m/s^2 (negative means braking).
        """
        v = vehicle.speed
        v0 = self.p.desired_speed
        a = self.p.max_accel
        delta = self.p.delta

        # Free-road acceleration: approaches 0 as v -> v0.
        # The exponent delta controls the shape; delta=4 gives a smooth
        # transition from full acceleration at low speed to coasting near v0.
        accel = a * (1.0 - (v / max(v0, 1e-6)) ** delta)

        if leader is not None:
            # Bumper-to-bumper gap to the leader
            s = _gap(leader, vehicle)
            s = max(s, 0.1)  # floor to avoid division-by-zero

            # Approach rate: positive when closing in on the leader
            dv = v - leader.speed

            # Desired dynamic following gap s* (Treiber et al. Eq. 2):
            #   s* = s0 + v*T + v*dv / (2*sqrt(a*b))
            #
            # - s0:   minimum gap at standstill
            # - v*T:  speed-dependent safe gap (time headway component)
            # - last term: additional gap when closing in (dv > 0)
            s_star = (self.p.min_gap
                      + v * self.p.time_headway
                      + v * dv / (2.0 * math.sqrt(a * self.p.comfortable_decel)))
            s_star = max(s_star, self.p.min_gap)  # clamp to at least s0

            # Interaction (braking) term: (s*/s)^2
            # When s < s*, this term dominates and produces deceleration.
            accel -= a * (s_star / s) ** 2

        return accel

    def get_action(self, vehicle, all_vehicles, num_lanes):
        """Compute IDM acceleration and keep the current lane.

        Args:
            vehicle:      The vehicle being controlled.
            all_vehicles: All vehicles in the scene.
            num_lanes:    Number of lanes (unused — IDM does not change lanes).

        Returns:
            ``(accel, current_lane)`` with accel clamped to
            ``[-1.5*b, a]`` where *b* is comfortable deceleration and
            *a* is maximum acceleration.
        """
        leader = _find_leader(vehicle, all_vehicles, vehicle.lane)
        accel = self.compute_accel(vehicle, leader)
        # Clamp: allow slightly harder braking than "comfortable" (1.5x)
        # but never exceed max acceleration
        accel = max(-self.p.comfortable_decel * 1.5, min(accel, self.p.max_accel))
        return accel, vehicle.lane


# ---------------------------------------------------------------------------
# MOBIL (Minimizing Overall Braking Induced by Lane changes)
# ---------------------------------------------------------------------------

@dataclass
class MOBILParams:
    """Parameters for the MOBIL lane-change decision model.

    Symbol mapping (Kesting et al. 2007):
        p       = politeness   — weight given to other drivers' disadvantage
        da_th   = threshold    — minimum net acceleration gain to trigger a
                                 lane change [m/s^2]
        b_safe  = safe_decel   — maximum braking imposed on the new follower
                                 by the lane change [m/s^2]
    """
    politeness: float = 0.5       # p      — altruism factor [0=selfish, 1=polite]
    threshold: float = 0.2        # da_th  — lane-change incentive threshold [m/s^2]
    safe_decel: float = 4.0       # b_safe — safety criterion for new follower [m/s^2]


class MOBILBehavior(TrafficBehavior):
    """MOBIL lane-change decision model layered on top of IDM.

    MOBIL decides *whether* to change lanes by comparing the acceleration
    benefit of moving to an adjacent lane against:
      - A **safety criterion**: the lane change must not force the new
        follower to brake harder than ``b_safe``.
      - An **incentive criterion**: the net benefit (own gain minus
        politeness-weighted cost to others) must exceed a threshold
        ``da_th``.

    Longitudinal control in every lane (current and candidate) is
    computed using the embedded IDM model.

    Attributes:
        idm:   ``IDMBehavior`` providing acceleration calculations.
        mobil: ``MOBILParams`` controlling the lane-change decision.
    """

    def __init__(self, idm_params: IDMParams = None,
                 mobil_params: MOBILParams = None):
        """Initialise MOBIL with IDM and MOBIL parameters.

        Args:
            idm_params:   IDM calibration (for longitudinal control).
            mobil_params: MOBIL decision thresholds.
        """
        self.idm = IDMBehavior(idm_params or IDMParams())
        self.mobil = mobil_params or MOBILParams()

    def get_action(self, vehicle, all_vehicles, num_lanes):
        """Evaluate MOBIL for each adjacent lane and return the best action.

        The algorithm:
            1. Compute IDM acceleration in the current lane (baseline).
            2. For each adjacent lane (left, right):
               a. Compute the ego's IDM acceleration in that lane.
               b. **Safety check**: compute the new follower's braking;
                  reject if it exceeds ``b_safe``.
               c. **Incentive check**: compute the MOBIL gain:
                  gain = (a_new - a_cur)
                         + p * [(a'_new_follower - a'_old_follower)
                                + (a'_old_follower_after - a'_old_follower_before)]
                  Accept if gain > threshold AND a_new > best so far.
            3. Use IDM accel for whichever lane was chosen.

        Args:
            vehicle:      The NPC vehicle being controlled.
            all_vehicles: All vehicles in the scene.
            num_lanes:    Number of lanes on the road.

        Returns:
            ``(accel, target_lane)`` — the best lane and corresponding
            IDM acceleration, clamped to safe bounds.
        """
        # Baseline: IDM acceleration in the current lane
        leader_cur = _find_leader(vehicle, all_vehicles, vehicle.lane)
        accel_cur = self.idm.compute_accel(vehicle, leader_cur)

        best_lane = vehicle.lane
        best_accel = accel_cur

        # Evaluate each adjacent lane
        for target_lane in [vehicle.lane - 1, vehicle.lane + 1]:
            # Skip out-of-bounds lanes
            if target_lane < 0 or target_lane >= num_lanes:
                continue

            # --- Ego's acceleration in the candidate lane ---
            leader_new = _find_leader(vehicle, all_vehicles, target_lane)
            accel_new = self.idm.compute_accel(vehicle, leader_new)

            # --- Safety criterion ---
            # The new follower (vehicle behind us in the target lane) must
            # not be forced to brake harder than b_safe.
            follower_new = _find_follower(vehicle, all_vehicles, target_lane)
            if follower_new is not None:
                # What would the new follower's accel be if we appeared
                # in front of them?
                accel_follower_after = self.idm.compute_accel(follower_new, vehicle)
                if accel_follower_after < -self.mobil.safe_decel:
                    continue  # UNSAFE — would force excessive braking

                # New follower's *current* acceleration (without us)
                follower_leader_cur = _find_leader(follower_new, all_vehicles,
                                                   target_lane)
                accel_follower_before = self.idm.compute_accel(
                    follower_new, follower_leader_cur)
            else:
                # No new follower — no safety concern and no cost to others
                accel_follower_before = 0.0
                accel_follower_after = 0.0

            # --- Old follower benefit ---
            # The vehicle currently behind us in our lane benefits when we
            # leave, because their leader changes from us to whoever is
            # ahead of us.
            follower_old = _find_follower(vehicle, all_vehicles, vehicle.lane)
            if follower_old is not None:
                old_follower_leader = _find_leader(follower_old, all_vehicles,
                                                   vehicle.lane)
                # Old follower's accel with us as their leader (before)
                accel_old_before = self.idm.compute_accel(
                    follower_old, vehicle)
                # Old follower's accel after we leave (new leader ahead)
                accel_old_after = self.idm.compute_accel(
                    follower_old, old_follower_leader)
            else:
                accel_old_before = 0.0
                accel_old_after = 0.0

            # --- MOBIL incentive criterion ---
            # gain = (own improvement)
            #        + p * (new follower's cost + old follower's benefit)
            #
            # Kesting et al. (2007) Eq. 3:
            #   a_tilde - a_c + p * [a'_n_after - a'_n_before
            #                        + a'_o_after - a'_o_before] > da_th
            gain = (accel_new - accel_cur
                    + self.mobil.politeness * (
                        (accel_follower_after - accel_follower_before)
                        + (accel_old_after - accel_old_before)))

            if gain > self.mobil.threshold and accel_new > best_accel:
                best_lane = target_lane
                best_accel = accel_new

        # Compute final IDM acceleration for the chosen lane
        if best_lane != vehicle.lane:
            leader_target = _find_leader(vehicle, all_vehicles, best_lane)
            final_accel = self.idm.compute_accel(vehicle, leader_target)
        else:
            final_accel = accel_cur

        # Clamp acceleration to safe physical bounds
        final_accel = max(-self.idm.p.comfortable_decel * 1.5,
                          min(final_accel, self.idm.p.max_accel))
        return final_accel, best_lane
