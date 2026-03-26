"""Main simulator: gym-like step/reset interface."""
from typing import Tuple, Optional
import numpy as np

from utils.config import Config, DEFAULT_CONFIG
from utils.types import VehicleState, Action, Observation, LateralAction
from environment.road import Road
from environment.vehicle import VehicleDynamics
from environment.traffic_manager import TrafficManager
from environment.observation import ObservationBuilder


class Simulator:
    """Self-driving simulator with multi-lane road and traffic.

    Interface:
        obs = sim.reset()
        obs, reward, done, info = sim.step(action)
    """

    def __init__(self, config: Config = None, seed: int = 42):
        self.config = config or DEFAULT_CONFIG
        self.road = Road(self.config.road)
        self.dynamics = VehicleDynamics(self.config.vehicle)
        self.traffic = TrafficManager(
            self.road, self.config.traffic, self.config.vehicle, seed=seed)
        self.obs_builder = ObservationBuilder(self.config.obs)
        self.rng = np.random.RandomState(seed)

        # State
        self.ego: Optional[VehicleState] = None
        self.npc_states = []
        self.step_count = 0
        self.prev_speed = 0.0
        self._done = False

    def reset(self) -> np.ndarray:
        """Reset episode. Returns observation vector."""
        self.step_count = 0
        self._done = False

        # Ego starts in a random lane, middle of road, at a reasonable speed
        lane = self.rng.randint(0, self.road.num_lanes)
        self.ego = VehicleState(
            x=self.road.road_length * 0.3,
            lane=lane,
            speed=20.0,
            vehicle_id=0,
        )
        self.prev_speed = self.ego.speed

        # Spawn NPCs
        self.npc_states = self.traffic.reset(self.ego.x, self.ego.speed)

        obs = self.obs_builder.build(self.ego, self.npc_states)
        return obs.to_vector()

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, dict]:
        """Take one simulation step.

        Returns:
            obs: observation vector (np.ndarray)
            reward: scalar reward
            done: whether episode ended
            info: dict with extra data
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        self.step_count += 1
        dt = self.config.sim.dt

        # Store previous state for reward computation
        prev_ego = self.ego

        # Step ego
        self.ego = self.dynamics.step(
            self.ego, action, dt, self.road.num_lanes)

        # Step traffic
        self.npc_states = self.traffic.step(self.ego, dt)

        # Check collisions
        collision = self._check_collision()

        # Compute reward
        reward = self._compute_reward(prev_ego, action, collision)

        # Check done
        done = False
        info = {"collision": collision, "step": self.step_count}
        if collision:
            done = True
        if self.step_count >= self.config.sim.episode_steps:
            done = True
            info["timeout"] = True

        self._done = done
        self.prev_speed = self.ego.speed

        obs = self.obs_builder.build(self.ego, self.npc_states)
        return obs.to_vector(), reward, done, info

    def get_all_vehicle_states(self):
        """Return (ego_state, npc_states) for visualization / planning."""
        return self.ego, self.npc_states

    def clone_state(self):
        """Snapshot simulator state for planner rollouts."""
        import copy
        return {
            "ego": copy.deepcopy(self.ego),
            "npcs": copy.deepcopy(self.npc_states),
            "traffic_npcs": copy.deepcopy(self.traffic.npcs),
            "step_count": self.step_count,
            "prev_speed": self.prev_speed,
            "done": self._done,
        }

    def restore_state(self, snapshot):
        """Restore simulator from a snapshot."""
        import copy
        self.ego = copy.deepcopy(snapshot["ego"])
        self.npc_states = copy.deepcopy(snapshot["npcs"])
        self.traffic.npcs = copy.deepcopy(snapshot["traffic_npcs"])
        self.step_count = snapshot["step_count"]
        self.prev_speed = snapshot["prev_speed"]
        self._done = snapshot["done"]

    # ------- internal -------

    def _check_collision(self) -> bool:
        """Check if ego collides with any NPC."""
        for npc in self.npc_states:
            if self._vehicles_collide(self.ego, npc):
                return True
        return False

    def _vehicles_collide(self, a: VehicleState, b: VehicleState) -> bool:
        """Simple axis-aligned bounding box collision."""
        # Must be in the same lane
        if a.lane != b.lane:
            return False
        # Longitudinal overlap
        half_len_a = a.length / 2
        half_len_b = b.length / 2
        gap = abs(a.x - b.x)
        return gap < (half_len_a + half_len_b)

    def _compute_reward(self, prev_ego: VehicleState,
                        action: Action, collision: bool) -> float:
        """Compute step reward."""
        reward = 0.0

        # Collision penalty
        if collision:
            reward += self.config.sim.collision_reward
            return reward

        # Speed reward: encourage driving near speed limit, not above
        speed_limit = max(self.config.road.speed_limit, 1e-6)
        if self.ego.speed <= speed_limit:
            reward += self.config.sim.speed_reward_scale * (self.ego.speed / speed_limit)
        else:
            # Above speed limit: reward drops sharply
            # At speed_limit: 1.0, at speed_limit+5: 0.0, beyond: negative
            excess = self.ego.speed - speed_limit
            reward += self.config.sim.speed_reward_scale * (1.0 - excess / 5.0)

        # Lane change penalty
        if action.lateral != LateralAction.KEEP:
            reward += self.config.sim.lane_change_penalty

        # Unsafe lane change penalty: if ego just changed lanes,
        # check gap to nearest car in the new lane (both ahead and behind).
        # Two-tier penalty:
        #   - Below hard_safe_gap (15m): full collision penalty (must never happen)
        #   - Between hard and soft (15-20m): graduated penalty (discourage marginal merges)
        # The soft zone is kept narrow to avoid excessive conservatism.
        if prev_ego.lane != self.ego.lane:
            hard_safe_gap_ahead = 15.0   # must have 15m ahead
            soft_safe_gap_ahead = 20.0
            hard_safe_gap_behind = 10.0  # behind is less dangerous — they brake
            soft_safe_gap_behind = 15.0

            for direction in ['behind', 'ahead']:
                if direction == 'behind':
                    candidates = [n for n in self.npc_states
                                  if n.lane == self.ego.lane and n.x < self.ego.x]
                    if candidates:
                        nearest = max(candidates, key=lambda n: n.x)
                        gap = self.ego.x - nearest.x
                    else:
                        continue
                    hard_safe = hard_safe_gap_behind
                    soft_safe = soft_safe_gap_behind
                else:
                    candidates = [n for n in self.npc_states
                                  if n.lane == self.ego.lane and n.x > self.ego.x]
                    if candidates:
                        nearest = min(candidates, key=lambda n: n.x)
                        gap = nearest.x - self.ego.x
                    else:
                        continue
                    hard_safe = hard_safe_gap_ahead
                    soft_safe = soft_safe_gap_ahead

                if gap < hard_safe:
                    reward += self.config.sim.collision_reward  # -100.0
                elif gap < soft_safe:
                    frac = (soft_safe - gap) / (soft_safe - hard_safe)
                    reward += self.config.sim.collision_reward * 0.15 * frac

        # Hard braking penalty
        accel = (self.ego.speed - prev_ego.speed) / self.config.sim.dt
        if accel < -self.config.sim.comfort_jerk_threshold:
            reward += self.config.sim.hard_brake_penalty

        return reward
