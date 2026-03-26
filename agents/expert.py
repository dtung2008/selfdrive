"""Rule-based expert driver for generating training data.

Strategy:
  1. Maintain safe following distance (IDM-like speed control).
  2. Change lanes to pass slow leaders when safe.
  3. Prefer a target lane (e.g., rightmost) when no benefit to being in left.
"""
import numpy as np
from agents.base import Agent
from utils.types import Action, LongitudinalAction, LateralAction, Observation
from utils.config import ObservationConfig


class ExpertAgent(Agent):
    """Deterministic rule-based expert."""

    def __init__(self, obs_config: ObservationConfig = None,
                 safe_distance: float = 20.0,
                 desired_speed: float = 30.0,
                 lane_change_gap: float = 15.0,
                 num_lanes: int = 2):
        self.obs_config = obs_config or ObservationConfig()
        self.safe_distance = safe_distance
        self.desired_speed = desired_speed
        self.lane_change_gap = lane_change_gap
        self.num_lanes = num_lanes
        self._step_count = 0
        self._last_lane_change_step = -100
        self._lane_change_cooldown = 10

    def reset(self):
        """Reset episode state."""
        self._step_count = 0
        self._last_lane_change_step = -100

    def act(self, obs: np.ndarray) -> Action:
        """Parse observation and decide action."""
        self._step_count += 1
        parsed = self._parse_obs(obs)
        ego_speed = parsed["ego_speed"]
        ego_lane = parsed["ego_lane"]
        neighbors = parsed["neighbors"]  # list of (rel_x, rel_lane, rel_speed, exists)

        # Find leader in current lane
        leader = self._find_closest_ahead(neighbors, lane_offset=0)

        # Longitudinal decision
        lon = self._longitudinal_decision(ego_speed, leader)

        # Lateral decision
        lat = self._lateral_decision(ego_speed, ego_lane, leader, neighbors)

        # Lane-change cooldown: prevent oscillation
        if lat != LateralAction.KEEP:
            steps_since = self._step_count - self._last_lane_change_step
            if steps_since < self._lane_change_cooldown:
                lat = LateralAction.KEEP
            else:
                self._last_lane_change_step = self._step_count

        return Action(longitudinal=lon, lateral=lat)

    def _parse_obs(self, obs: np.ndarray) -> dict:
        ego_speed = obs[0]
        ego_lane = int(round(obs[1]))
        ego_x = obs[2]
        k = self.obs_config.k_neighbors
        raw = obs[3:].reshape(k, 4)
        neighbors = []
        for i in range(k):
            if raw[i, 3] > 0.5:  # exists
                neighbors.append({
                    "rel_x": raw[i, 0],
                    "rel_lane": int(round(raw[i, 1])),
                    "rel_speed": raw[i, 2],
                })
        return {"ego_speed": ego_speed, "ego_lane": ego_lane,
                "ego_x": ego_x, "neighbors": neighbors}

    def _find_closest_ahead(self, neighbors, lane_offset=0):
        """Find closest vehicle ahead in lane ego_lane + lane_offset."""
        ahead = [n for n in neighbors
                 if n["rel_lane"] == lane_offset and n["rel_x"] > 0]
        if not ahead:
            return None
        return min(ahead, key=lambda n: n["rel_x"])

    def _find_closest_behind(self, neighbors, lane_offset=0):
        ahead = [n for n in neighbors
                 if n["rel_lane"] == lane_offset and n["rel_x"] < 0]
        if not ahead:
            return None
        return max(ahead, key=lambda n: n["rel_x"])

    def _longitudinal_decision(self, ego_speed, leader) -> LongitudinalAction:
        if leader is not None and leader["rel_x"] < self.safe_distance:
            # Too close — slow down
            if leader["rel_speed"] < 0:
                # Leader is slower
                return LongitudinalAction.DECELERATE
            else:
                return LongitudinalAction.KEEP
        elif ego_speed < self.desired_speed - 0.5:
            return LongitudinalAction.ACCELERATE
        elif ego_speed > self.desired_speed + 2.0:
            return LongitudinalAction.DECELERATE
        else:
            return LongitudinalAction.KEEP

    def _lateral_decision(self, ego_speed, ego_lane, leader,
                          neighbors) -> LateralAction:
        # If leader is slow and close, try to change lanes
        # Two triggers:
        # 1. Leader is actively slower (approaching)
        # 2. Stuck behind leader well below desired speed (already matched speed)
        leader_blocking = (leader is not None
                           and leader["rel_x"] < self.safe_distance * 1.5
                           and leader["rel_speed"] < -2.0)
        stuck_behind = (leader is not None
                        and leader["rel_x"] < self.safe_distance * 1.5
                        and ego_speed < self.desired_speed - 5.0)
        should_pass = leader_blocking or stuck_behind

        if not should_pass:
            return LateralAction.KEEP

        # Emergency: gap ahead < safe following distance (speed * 0.5)
        # At 30 m/s: emergency at < 15m. At 17 m/s: emergency at < 8.5m
        # Also trigger when stuck (speed well below desired with close leader)
        safe_following = ego_speed * 0.5 if ego_speed > 10 else 5.0
        emergency = (leader is not None and
                     (leader["rel_x"] < safe_following or
                      (stuck_behind and leader["rel_x"] < self.safe_distance)))

        # Check both directions, pick the one with more room (or random if equal)
        # Only consider lanes that have significantly more room than current lane
        import random
        current_gap = leader["rel_x"] if leader else float('inf')
        min_improvement = 10.0  # target lane must have at least 10m more room

        directions = [(-1, LateralAction.LEFT), (1, LateralAction.RIGHT)]
        safe_options = []
        for direction, lat_action in directions:
            target_lane = ego_lane + direction
            if target_lane < 0 or target_lane >= self.num_lanes:
                continue
            if self._lane_is_safe(neighbors, direction, emergency=emergency):
                # Find ahead gap in target lane
                ahead = self._find_closest_ahead(neighbors, direction)
                gap = ahead["rel_x"] if ahead else float('inf')
                # Only worth changing if target has meaningfully more room
                if gap > current_gap + min_improvement:
                    safe_options.append((gap, lat_action))

        if not safe_options:
            return LateralAction.KEEP
        if len(safe_options) == 1:
            return safe_options[0][1]
        # Pick the lane with more room ahead
        if abs(safe_options[0][0] - safe_options[1][0]) > 5.0:
            return max(safe_options, key=lambda x: x[0])[1]
        # Similar room — pick randomly
        return random.choice(safe_options)[1]

    def _lane_is_safe(self, neighbors, lane_offset, emergency=False) -> bool:
        """Check if adjacent lane has sufficient gap.

        Normal mode: ahead > 20m, behind > 15m (conservative)
        Emergency mode: ahead > 10m, behind > 5m (must escape)
        """
        if emergency:
            min_gap_ahead = 10.0
            min_gap_behind = 5.0
        else:
            min_gap_ahead = max(self.lane_change_gap, 20.0)
            min_gap_behind = 15.0
        ahead = self._find_closest_ahead(neighbors, lane_offset)
        behind = self._find_closest_behind(neighbors, lane_offset)

        if ahead is not None and ahead["rel_x"] < min_gap_ahead:
            return False
        if behind is not None and abs(behind["rel_x"]) < min_gap_behind:
            return False
        return True
