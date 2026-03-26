"""Traffic manager: spawn, despawn, and step NPC vehicles."""
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
    """Manages NPC vehicles: spawning, behaviour assignment, stepping."""

    def __init__(self, road: Road,
                 traffic_config: TrafficConfig = None,
                 vehicle_config: VehicleConfig = None,
                 seed: int = 42):
        self.road = road
        self.tc = traffic_config or TrafficConfig()
        self.vc = vehicle_config or VehicleConfig()
        self.dynamics = VehicleDynamics(self.vc)
        self.rng = np.random.RandomState(seed)
        self._next_id = 1000  # NPC ids start at 1000

        # vehicle_id -> (VehicleState, TrafficBehavior)
        self.npcs: Dict[int, Tuple[VehicleState, TrafficBehavior]] = {}

    def reset(self, ego_x: float, ego_speed: float = 20.0) -> List[VehicleState]:
        """Spawn initial set of NPCs around ego position. Returns NPC states."""
        self.npcs.clear()
        self._next_id = 1000
        self._spawn_npcs(ego_x, ego_speed, self.tc.num_npcs, initial=True)
        return self.get_states()

    def step(self, ego_state: VehicleState, dt: float) -> List[VehicleState]:
        """Step all NPCs, despawn far ones, spawn new ones if needed."""
        all_vehicles = [ego_state] + self.get_states()

        # Step each NPC
        updated: Dict[int, Tuple[VehicleState, TrafficBehavior]] = {}
        for vid, (state, behavior) in self.npcs.items():
            accel, target_lane = behavior.get_action(
                state, all_vehicles, self.road.num_lanes)
            new_state = self.dynamics.step_with_acceleration(
                state, accel, target_lane, dt, self.road.num_lanes)
            updated[vid] = (new_state, behavior)
        self.npcs = updated

        # Despawn far vehicles
        self._despawn(ego_state.x)

        # Respawn to maintain target count (only ahead of ego)
        deficit = self.tc.num_npcs - len(self.npcs)
        if deficit > 0:
            self._spawn_npcs(ego_state.x, ego_state.speed, deficit, initial=False)

        return self.get_states()

    def get_states(self) -> List[VehicleState]:
        """Return list of current NPC states."""
        return [state for state, _ in self.npcs.values()]

    # ------- internal -------

    def _spawn_npcs(self, ego_x: float, ego_speed: float, count: int,
                    initial: bool = True):
        """Spawn `count` NPCs.

        Args:
            ego_x: Ego longitudinal position.
            ego_speed: Ego speed for setting NPC spawn speed.
            count: Number of NPCs to spawn.
            initial: If True (reset), spawn around ego (ahead and behind).
                     If False (respawn), only spawn ahead and outside view window.
        """
        existing_positions = [(s.x, s.lane) for s in self.get_states()]

        # Spawn range: initial allows behind ego, respawn only ahead
        if initial:
            x_min = ego_x - self.tc.spawn_range * 0.3
            x_max = ego_x + self.tc.spawn_range
        else:
            # Respawn: only ahead, outside visible range (~100m ahead)
            x_min = ego_x + 100.0
            x_max = ego_x + self.tc.spawn_range

        spawned = 0
        attempts = 0
        while spawned < count and attempts < count * 10:
            attempts += 1
            x = self.rng.uniform(x_min, x_max)
            lane = self.rng.randint(0, self.road.num_lanes)

            # Spawn speed near ego speed
            margin = 3.0
            speed_lo = max(5.0, ego_speed - margin)
            speed_hi = min(self.vc.max_speed, ego_speed + margin)
            speed = self.rng.uniform(speed_lo, speed_hi)

            # Check minimum gap to existing vehicles
            too_close = any(abs(x - ex) < self.tc.min_spawn_gap and lane == el
                           for ex, el in existing_positions)
            if abs(x - ego_x) < self.tc.min_spawn_gap:
                too_close = True

            if too_close:
                continue

            vid = self._next_id
            self._next_id += 1
            state = VehicleState(x=x, lane=lane, speed=speed, vehicle_id=vid)
            behavior = self._assign_behavior(speed)
            self.npcs[vid] = (state, behavior)
            existing_positions.append((x, lane))
            spawned += 1

    def _despawn(self, ego_x: float):
        """Remove NPCs too far from ego."""
        to_remove = [vid for vid, (state, _) in self.npcs.items()
                     if abs(state.x - ego_x) > self.tc.despawn_distance]
        for vid in to_remove:
            del self.npcs[vid]

    def _assign_behavior(self, speed: float) -> TrafficBehavior:
        """Assign IDM-based behaviour to an NPC.

        All NPCs use IDM car-following. In multi-lane mode, MOBIL
        lane-change logic is added on top based on behavior_mix.

        The spawn speed is used as the desired speed so NPCs maintain
        roughly the speed they were created with.

        Args:
            speed: The spawn speed, used as IDM desired_speed.
        """
        mix = np.array(self.tc.behavior_mix)
        mix = mix / mix.sum()
        choice = self.rng.choice(len(mix), p=mix)

        idm_params = IDMParams(
            desired_speed=speed,
            time_headway=self.tc.idm_time_headway,
            min_gap=self.tc.idm_min_gap,
            max_accel=self.tc.idm_accel,
            comfortable_decel=self.tc.idm_decel,
            delta=self.tc.idm_delta,
        )

        if choice <= 1:
            # IDM only (covers both old "constant" and "idm" choices)
            return IDMBehavior(idm_params)
        else:
            # IDM + MOBIL
            mobil_params = MOBILParams(
                politeness=self.tc.mobil_politeness,
                threshold=self.tc.mobil_threshold,
                safe_decel=self.tc.mobil_safe_decel,
            )
            return MOBILBehavior(idm_params, mobil_params)
