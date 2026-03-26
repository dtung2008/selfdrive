"""Traffic manager: spawn, despawn, and step NPC vehicles.

This module is responsible for the full lifecycle of non-player-controlled
(NPC) vehicles within the simulation:

    1. **Spawning** — placing new NPCs on the road with a random position,
       lane, speed, and driving behaviour.
    2. **Stepping** — advancing every NPC by one timestep using its assigned
       behaviour (IDM or MOBIL) and the shared ``VehicleDynamics`` model.
    3. **Despawning** — removing NPCs that have drifted too far from the ego
       vehicle (beyond ``despawn_distance``).
    4. **Respawning** — topping up the NPC population to ``num_npcs`` by
       creating fresh vehicles ahead of the ego, outside the visible range,
       so they appear to flow naturally into the scene.

The manager also assigns each NPC a driving behaviour drawn from a
configurable mixture distribution (``behavior_mix``).
"""
from typing import Dict, List, Tuple
import numpy as np

from utils.types import VehicleState
from utils.config import TrafficConfig, VehicleConfig
from environment.road import Road
from environment.vehicle import VehicleDynamics
from environment.traffic_behaviors import (
    TrafficBehavior, ConstantSpeedBehavior,
    IDMBehavior, IDMParams, MOBILBehavior, MOBILParams,
)


class TrafficManager:
    """Manages the full lifecycle of NPC vehicles in the simulation.

    Each NPC is stored as a ``(VehicleState, TrafficBehavior)`` pair keyed
    by its unique integer ``vehicle_id``.  The manager exposes a ``reset``
    method for episode initialisation and a ``step`` method that is called
    once per simulation timestep.

    Attributes:
        road:      Reference to the ``Road`` geometry.
        tc:        ``TrafficConfig`` with spawn/despawn parameters.
        vc:        ``VehicleConfig`` with speed/accel limits.
        dynamics:  ``VehicleDynamics`` instance shared by all NPCs.
        rng:       Seeded numpy random state for reproducibility.
        _next_id:  Auto-incrementing counter for NPC vehicle IDs (starting
                   at 1000 to avoid collision with ego ID 0).
        npcs:      Dict mapping ``vehicle_id`` to ``(VehicleState,
                   TrafficBehavior)`` for every active NPC.
    """

    def __init__(self, road: Road,
                 traffic_config: TrafficConfig = None,
                 vehicle_config: VehicleConfig = None,
                 seed: int = 42):
        """Initialise the traffic manager.

        Args:
            road:           Road geometry (needed for lane count).
            traffic_config: NPC population and behaviour settings.
            vehicle_config: Vehicle dynamics limits (shared with NPCs).
            seed:           Random seed for spawn randomisation.
        """
        self.road = road
        self.tc = traffic_config or TrafficConfig()
        self.vc = vehicle_config or VehicleConfig()
        self.dynamics = VehicleDynamics(self.vc)
        self.rng = np.random.RandomState(seed)
        self._next_id = 1000  # NPC IDs start at 1000 to avoid clash with ego (ID 0)

        # Active NPCs: vehicle_id -> (VehicleState, TrafficBehavior)
        self.npcs: Dict[int, Tuple[VehicleState, TrafficBehavior]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, ego_x: float, ego_speed: float = 20.0) -> List[VehicleState]:
        """Clear all NPCs and spawn a fresh population for a new episode.

        Args:
            ego_x:     Ego's longitudinal position (spawn centre).
            ego_speed: Ego's speed, used to set NPC spawn speeds nearby.

        Returns:
            A list of ``VehicleState`` objects for every spawned NPC.
        """
        self.npcs.clear()
        self._next_id = 1000
        self._spawn_npcs(ego_x, ego_speed, self.tc.num_npcs, initial=True)
        return self.get_states()

    def step(self, ego_state: VehicleState, dt: float) -> List[VehicleState]:
        """Advance all NPCs by one timestep, then despawn/respawn as needed.

        Execution order:
            1. Build a combined vehicle list (ego + all NPCs) so that each
               NPC's behaviour can see the full traffic picture.
            2. Compute each NPC's acceleration and target lane via its
               assigned behaviour (IDM/MOBIL).
            3. Apply kinematic updates via ``VehicleDynamics``.
            4. Despawn NPCs that are too far from the ego.
            5. Respawn new NPCs if the population dropped below target.

        Args:
            ego_state: The ego vehicle's current state (needed for
                       behaviour calculations and spawn/despawn logic).
            dt:        Simulation timestep in seconds.

        Returns:
            Updated list of ``VehicleState`` objects for all active NPCs.
        """
        # Combine ego + NPCs so behaviours can see all vehicles
        all_vehicles = [ego_state] + self.get_states()

        # Step each NPC through its behaviour and dynamics
        updated: Dict[int, Tuple[VehicleState, TrafficBehavior]] = {}
        for vid, (state, behavior) in self.npcs.items():
            accel, target_lane = behavior.get_action(
                state, all_vehicles, self.road.num_lanes)
            new_state = self.dynamics.step_with_acceleration(
                state, accel, target_lane, dt, self.road.num_lanes)
            updated[vid] = (new_state, behavior)
        self.npcs = updated

        # Despawn NPCs that have drifted too far from the ego
        self._despawn(ego_state.x)

        # Respawn to maintain the target NPC count (new vehicles ahead)
        deficit = self.tc.num_npcs - len(self.npcs)
        if deficit > 0:
            self._spawn_npcs(ego_state.x, ego_state.speed, deficit, initial=False)

        return self.get_states()

    def get_states(self) -> List[VehicleState]:
        """Return a flat list of all active NPC vehicle states.

        Returns:
            List of ``VehicleState`` objects (one per active NPC).
        """
        return [state for state, _ in self.npcs.values()]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn_npcs(self, ego_x: float, ego_speed: float, count: int,
                    initial: bool = True):
        """Spawn ``count`` new NPC vehicles on the road.

        During episode reset (``initial=True``), NPCs are placed both
        ahead of and behind the ego to populate the scene realistically.
        During mid-episode respawning (``initial=False``), NPCs are only
        placed *ahead* of the ego and beyond the visible window (~100 m)
        so they appear to drive into view naturally.

        A rejection-sampling loop ensures a minimum gap between all
        vehicles (including the ego) to prevent overlapping spawns.

        Args:
            ego_x:     Ego's longitudinal position.
            ego_speed: Ego's speed (NPC speed is randomised near this value).
            count:     Number of NPCs to attempt to spawn.
            initial:   If *True*, spread NPCs around the ego (reset).
                       If *False*, only spawn ahead of the ego (respawn).
        """
        existing_positions = [(s.x, s.lane) for s in self.get_states()]

        # Determine the longitudinal spawn window
        if initial:
            # Reset: spread NPCs from 30% behind ego to full range ahead
            x_min = ego_x - self.tc.spawn_range * 0.3
            x_max = ego_x + self.tc.spawn_range
        else:
            # Respawn: only ahead, starting 100 m beyond ego to stay out of
            # the immediate observation window
            x_min = ego_x + 100.0
            x_max = ego_x + self.tc.spawn_range

        spawned = 0
        attempts = 0
        # Rejection sampling with a generous attempt budget (10x target)
        while spawned < count and attempts < count * 10:
            attempts += 1

            # Random position and lane
            x = self.rng.uniform(x_min, x_max)
            lane = self.rng.randint(0, self.road.num_lanes)

            # Random speed near ego speed (within +/- margin), clamped to
            # [5, max_speed] so NPCs are never too slow or too fast
            margin = 3.0
            speed_lo = max(5.0, ego_speed - margin)
            speed_hi = min(self.vc.max_speed, ego_speed + margin)
            speed = self.rng.uniform(speed_lo, speed_hi)

            # Reject if too close to any existing vehicle in the same lane
            too_close = any(abs(x - ex) < self.tc.min_spawn_gap and lane == el
                           for ex, el in existing_positions)
            # Also reject if too close to the ego itself
            if abs(x - ego_x) < self.tc.min_spawn_gap:
                too_close = True

            if too_close:
                continue

            # Create and register the new NPC
            vid = self._next_id
            self._next_id += 1
            state = VehicleState(x=x, lane=lane, speed=speed, vehicle_id=vid)
            behavior = self._assign_behavior(speed)
            self.npcs[vid] = (state, behavior)
            existing_positions.append((x, lane))
            spawned += 1

    def _despawn(self, ego_x: float):
        """Remove NPC vehicles that are too far from the ego.

        Any NPC whose longitudinal distance from the ego exceeds
        ``despawn_distance`` (from config) is deleted.  This keeps the
        active NPC pool bounded and focused around the ego.

        Args:
            ego_x: Ego's current longitudinal position.
        """
        to_remove = [vid for vid, (state, _) in self.npcs.items()
                     if abs(state.x - ego_x) > self.tc.despawn_distance]
        for vid in to_remove:
            del self.npcs[vid]

    def _assign_behavior(self, speed: float) -> TrafficBehavior:
        """Assign a driving behaviour to a newly spawned NPC.

        All NPCs use IDM-based car-following.  A configurable mixture
        (``behavior_mix``) determines whether the NPC also gets MOBIL
        lane-change logic:
            - Indices 0 and 1 in the mix -> IDM only (no lane changes).
            - Index 2 in the mix        -> IDM + MOBIL (lane changes).

        The NPC's spawn speed is used as the IDM desired speed so that
        each NPC roughly maintains the speed it was created with.

        Args:
            speed: The NPC's spawn speed, used as the IDM ``desired_speed``.

        Returns:
            A ``TrafficBehavior`` instance (either ``IDMBehavior`` or
            ``MOBILBehavior``).
        """
        # Normalise the behaviour mix to a probability distribution
        mix = np.array(self.tc.behavior_mix)
        mix = mix / mix.sum()
        choice = self.rng.choice(len(mix), p=mix)

        # Build IDM parameters from traffic config, using spawn speed as
        # the desired free-flow speed
        idm_params = IDMParams(
            desired_speed=speed,
            time_headway=self.tc.idm_time_headway,
            min_gap=self.tc.idm_min_gap,
            max_accel=self.tc.idm_accel,
            comfortable_decel=self.tc.idm_decel,
            delta=self.tc.idm_delta,
        )

        if choice <= 1:
            # IDM only — car-following without lane changes
            # (covers both old "constant" and "idm" slots in the mix)
            return IDMBehavior(idm_params)
        else:
            # IDM + MOBIL — car-following with lane-change decisions
            mobil_params = MOBILParams(
                politeness=self.tc.mobil_politeness,
                threshold=self.tc.mobil_threshold,
                safe_decel=self.tc.mobil_safe_decel,
            )
            return MOBILBehavior(idm_params, mobil_params)
