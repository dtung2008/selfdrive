"""Main simulator: gym-like step/reset interface.

This module ties together every component of the driving environment —
road geometry, vehicle dynamics, NPC traffic, observation building, and
reward computation — into a single ``Simulator`` class that exposes the
standard ``reset() / step(action)`` loop used by reinforcement-learning
agents and planning algorithms.

Lifecycle:
    1. ``obs = sim.reset()``       — start a new episode
    2. ``obs, r, done, info = sim.step(action)``  — advance one timestep
    3. Repeat step 2 until ``done`` is *True*, then go back to step 1.

The simulator also supports ``clone_state()`` / ``restore_state()`` for
tree-search planners that need to branch and roll back.
"""
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

    The simulator manages an *ego* vehicle controlled by the agent and a
    population of *NPC* vehicles managed by the ``TrafficManager``.  At each
    timestep the ego executes a discrete ``Action`` (accelerate / decelerate /
    maintain speed combined with left / right / keep lane), the NPCs follow
    their IDM/MOBIL behaviours, collisions are detected, and a scalar reward
    is returned.

    Attributes:
        config:      Full configuration tree (road, vehicle, traffic, sim, obs).
        road:        ``Road`` instance describing geometry.
        dynamics:    ``VehicleDynamics`` for the ego vehicle.
        traffic:     ``TrafficManager`` handling all NPCs.
        obs_builder: ``ObservationBuilder`` producing fixed-size observations.
        rng:         Numpy random state seeded for reproducibility.
        ego:         Current ego ``VehicleState`` (None before first reset).
        npc_states:  List of current NPC ``VehicleState`` objects.
        step_count:  Number of steps taken in the current episode.
        prev_speed:  Ego speed at the *previous* timestep (for jerk / reward).
        _done:       Whether the current episode has terminated.

    Interface:
        obs = sim.reset()
        obs, reward, done, info = sim.step(action)
    """

    def __init__(self, config: Config = None, seed: int = 42):
        """Initialise the simulator and all sub-components.

        Args:
            config: Full configuration object.  Falls back to
                    ``DEFAULT_CONFIG`` when *None*.
            seed:   Random seed for reproducibility (ego spawn + traffic).
        """
        self.config = config or DEFAULT_CONFIG
        self.road = Road(self.config.road)
        self.dynamics = VehicleDynamics(self.config.vehicle)
        self.traffic = TrafficManager(
            self.road, self.config.traffic, self.config.vehicle, seed=seed)
        self.obs_builder = ObservationBuilder(self.config.obs)
        self.rng = np.random.RandomState(seed)

        # Episode state (populated on reset)
        self.ego: Optional[VehicleState] = None
        self.npc_states = []
        self.step_count = 0
        self.prev_speed = 0.0
        self._done = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the episode and return the initial observation vector.

        The ego vehicle is placed at 30 % of the road length in a random
        lane with an initial speed of 20 m/s.  NPC traffic is spawned
        around the ego.

        Returns:
            A 1-D numpy float32 array representing the initial observation.
        """
        self.step_count = 0
        self._done = False

        # Ego starts in a random lane, at 30% of road length, at 20 m/s
        lane = self.rng.randint(0, self.road.num_lanes)
        self.ego = VehicleState(
            x=self.road.road_length * 0.3,
            lane=lane,
            speed=20.0,
            vehicle_id=0,
        )
        self.prev_speed = self.ego.speed

        # Spawn NPC traffic around the ego
        self.npc_states = self.traffic.reset(self.ego.x, self.ego.speed)

        obs = self.obs_builder.build(self.ego, self.npc_states)
        return obs.to_vector()

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, dict]:
        """Advance the simulation by one timestep.

        Execution order:
            1. Apply ``action`` to the ego vehicle (kinematics).
            2. Step all NPC traffic (IDM/MOBIL + despawn/respawn).
            3. Check for ego-NPC collisions.
            4. Compute the scalar reward.
            5. Determine whether the episode is done (collision or timeout).

        Args:
            action: The ego vehicle's ``Action`` for this timestep.

        Returns:
            A 4-tuple ``(obs, reward, done, info)`` where:
                - *obs*    is a 1-D numpy float32 observation vector.
                - *reward* is a scalar float.
                - *done*   is True if the episode has ended.
                - *info*   is a dict with keys ``"collision"`` (bool),
                  ``"step"`` (int), and optionally ``"timeout"`` (bool).

        Raises:
            RuntimeError: If called after the episode has already ended
                          without an intervening ``reset()``.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        self.step_count += 1
        dt = self.config.sim.dt

        # Snapshot ego before the update for reward computation
        prev_ego = self.ego

        # 1. Update ego state via kinematic model
        self.ego = self.dynamics.step(
            self.ego, action, dt, self.road.num_lanes)

        # 2. Update NPC traffic (behaviours + spawn/despawn)
        self.npc_states = self.traffic.step(self.ego, dt)

        # 3. Collision detection (ego vs every NPC)
        collision = self._check_collision()

        # 4. Reward calculation (uses previous ego state for delta-based terms)
        reward = self._compute_reward(prev_ego, action, collision)

        # 5. Termination check: collision or max-step timeout
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
        """Return ``(ego_state, npc_states)`` for visualisation or planning.

        Returns:
            A 2-tuple of the ego ``VehicleState`` and a list of NPC
            ``VehicleState`` objects.
        """
        return self.ego, self.npc_states

    def clone_state(self):
        """Create a deep-copy snapshot of the full simulator state.

        This is used by tree-search planners (e.g. MCTS) that need to
        branch from the current state, roll out different action
        sequences, and later restore the original state.

        Returns:
            A dict containing deep copies of every mutable field:
            ego, NPCs, traffic internals, step count, and done flag.
        """
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
        """Restore the simulator from a previously cloned snapshot.

        Args:
            snapshot: A dict returned by ``clone_state()``.
        """
        import copy
        self.ego = copy.deepcopy(snapshot["ego"])
        self.npc_states = copy.deepcopy(snapshot["npcs"])
        self.traffic.npcs = copy.deepcopy(snapshot["traffic_npcs"])
        self.step_count = snapshot["step_count"]
        self.prev_speed = snapshot["prev_speed"]
        self._done = snapshot["done"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_collision(self) -> bool:
        """Check whether the ego vehicle collides with any NPC.

        Iterates over every NPC and tests axis-aligned bounding-box
        overlap with the ego.

        Returns:
            *True* if at least one collision is detected.
        """
        for npc in self.npc_states:
            if self._vehicles_collide(self.ego, npc):
                return True
        return False

    def _vehicles_collide(self, a: VehicleState, b: VehicleState) -> bool:
        """Test axis-aligned bounding-box collision between two vehicles.

        Because lateral movement is discrete (vehicles occupy exactly one
        lane), the lateral check reduces to a simple lane-equality test.
        Longitudinally, the vehicles collide when the centre-to-centre
        distance is less than the sum of their half-lengths (i.e. their
        bounding boxes overlap).

        Args:
            a: First vehicle state.
            b: Second vehicle state.

        Returns:
            *True* if the two vehicles overlap.
        """
        # Vehicles in different lanes cannot collide in this discrete model
        if a.lane != b.lane:
            return False
        # Longitudinal overlap: gap between centres < sum of half-lengths
        half_len_a = a.length / 2
        half_len_b = b.length / 2
        gap = abs(a.x - b.x)
        return gap < (half_len_a + half_len_b)

    def _compute_reward(self, prev_ego: VehicleState,
                        action: Action, collision: bool) -> float:
        """Compute the scalar reward for the current transition.

        The reward is a weighted sum of several terms designed to encourage
        safe, efficient, and comfortable driving:

        1. **Collision penalty** — a large negative constant that
           immediately ends the episode.
        2. **Speed reward** — encourages driving near the speed limit.
           Below the limit the reward scales linearly from 0 to
           ``speed_reward_scale``.  Above the limit it decreases linearly,
           reaching zero at ``speed_limit + 5 m/s`` and going negative
           beyond that.
        3. **Lane-change penalty** — a small negative cost for every lane
           change to discourage unnecessary weaving.
        4. **Unsafe lane-change penalty** — a two-tier penalty applied when
           the ego changes lanes and the gap to the nearest vehicle in the
           new lane is dangerously small:
             - *Hard zone* (gap < hard_safe_gap): full collision-level
               penalty — this should never happen.
             - *Soft zone* (hard_safe_gap <= gap < soft_safe_gap): a
               graduated fraction of the collision penalty, discouraging
               marginal merges.
           Separate thresholds are used for vehicles ahead vs. behind
           because a trailing vehicle can brake to create space.
        5. **Hard-braking penalty** — penalises decelerations that exceed
           a comfort threshold to encourage smooth driving.

        Args:
            prev_ego:  Ego state *before* the current step.
            action:    The action the ego took this step.
            collision: Whether a collision was detected this step.

        Returns:
            A scalar float reward.
        """
        reward = 0.0

        # ---- 1. Collision penalty (terminates episode) ----
        if collision:
            reward += self.config.sim.collision_reward
            return reward  # no other terms matter when crashed

        # ---- 2. Speed reward ----
        # Normalise ego speed by the speed limit for a [0, 1] base reward.
        speed_limit = max(self.config.road.speed_limit, 1e-6)  # avoid /0
        if self.ego.speed <= speed_limit:
            # Linear ramp: 0 at standstill, speed_reward_scale at limit
            reward += self.config.sim.speed_reward_scale * (self.ego.speed / speed_limit)
        else:
            # Above the limit the reward drops linearly.
            # At speed_limit      -> scale * 1.0
            # At speed_limit + 5  -> scale * 0.0
            # Beyond              -> negative (penalty)
            excess = self.ego.speed - speed_limit
            reward += self.config.sim.speed_reward_scale * (1.0 - excess / 5.0)

        # ---- 3. Lane-change penalty ----
        if action.lateral != LateralAction.KEEP:
            reward += self.config.sim.lane_change_penalty

        # ---- 4. Unsafe lane-change penalty (two-tier) ----
        # Only evaluated when the ego actually moved to a new lane.
        if prev_ego.lane != self.ego.lane:
            # Gap thresholds (metres).  The "hard" zone is an almost-
            # certain collision zone; the "soft" zone is a graduated
            # caution zone between hard and soft limits.
            hard_safe_gap_ahead = 15.0   # minimum acceptable gap ahead
            soft_safe_gap_ahead = 20.0
            hard_safe_gap_behind = 10.0  # behind thresholds are smaller
            soft_safe_gap_behind = 15.0  # because the trailing car can brake

            for direction in ['behind', 'ahead']:
                if direction == 'behind':
                    # Find the nearest NPC *behind* ego in the new lane
                    candidates = [n for n in self.npc_states
                                  if n.lane == self.ego.lane and n.x < self.ego.x]
                    if candidates:
                        nearest = max(candidates, key=lambda n: n.x)
                        gap = self.ego.x - nearest.x
                    else:
                        continue  # no vehicle behind — no penalty
                    hard_safe = hard_safe_gap_behind
                    soft_safe = soft_safe_gap_behind
                else:
                    # Find the nearest NPC *ahead* of ego in the new lane
                    candidates = [n for n in self.npc_states
                                  if n.lane == self.ego.lane and n.x > self.ego.x]
                    if candidates:
                        nearest = min(candidates, key=lambda n: n.x)
                        gap = nearest.x - self.ego.x
                    else:
                        continue  # no vehicle ahead — no penalty
                    hard_safe = hard_safe_gap_ahead
                    soft_safe = soft_safe_gap_ahead

                if gap < hard_safe:
                    # Hard zone: full collision-level penalty
                    reward += self.config.sim.collision_reward  # e.g. -100.0
                elif gap < soft_safe:
                    # Soft zone: linearly interpolated fraction of the
                    # collision penalty.  At hard_safe -> 15 % of penalty;
                    # at soft_safe -> 0 %.
                    frac = (soft_safe - gap) / (soft_safe - hard_safe)
                    reward += self.config.sim.collision_reward * 0.15 * frac

        # ---- 5. Hard-braking penalty ----
        # Compute instantaneous acceleration from speed change.
        accel = (self.ego.speed - prev_ego.speed) / self.config.sim.dt
        if accel < -self.config.sim.comfort_jerk_threshold:
            reward += self.config.sim.hard_brake_penalty

        return reward
