"""Rule-based expert driver for generating imitation-learning training data.

This module implements a deterministic, hand-crafted driving policy that
mimics reasonable human driving behaviour on a multi-lane highway.  It is
used to collect (observation, action) pairs that a neural-network policy
can later learn to reproduce via behavioural cloning.

High-level strategy
-------------------
1. **Longitudinal control** -- Maintain a safe following distance behind the
   lead vehicle using physics-based stopping distance (IDM-inspired).
   Anticipate braking cascades from the 2nd leader's acceleration.
   Accelerate toward a desired cruise speed when the road ahead is clear;
   decelerate when the gap to the leader becomes unsafe.
2. **Lane-change decisions** -- Attempt to overtake a slow leader by
   switching to an adjacent lane when there is a sufficient gap both ahead
   and behind in the target lane.  An *emergency* mode relaxes the gap
   thresholds when the ego vehicle is critically close to the leader.
   Uses continuous urgency scoring rather than a binary motivation gate.
3. **Lane preference** -- When urgency is low, the agent holds its current
   lane (no gratuitous weaving).
4. **Same-lane behind awareness** -- A fast approaching tailgater adds
   urgency to lane-change decisions.

A cooldown mechanism prevents rapid back-and-forth lane oscillations.
"""

import numpy as np
from agents.base import Agent
from utils.types import Action, LongitudinalAction, LateralAction, Observation
from utils.config import ObservationConfig, ExpertConfig


class ExpertAgent(Agent):
    """Deterministic rule-based expert driver with acceleration awareness.

    The agent reads a flat observation vector produced by the simulator,
    parses it into ego state and neighbor information (including
    acceleration), then applies a hierarchy of rules to decide on a
    longitudinal and lateral action each time step.

    Key improvements over a naive rule-based driver:
    - Physics-based stopping distance accounts for closing speed.
    - Cascade braking: anticipates 1st leader braking by observing 2nd
      leader's deceleration.
    - Stable gap: allows acceleration when leader is pulling away.
    - Same-lane behind awareness: tailgater pressure adds lane-change urgency.
    - Continuous urgency replaces binary motivation gate.
    """

    def __init__(self, obs_config: ObservationConfig = None,
                 expert_config: ExpertConfig = None,
                 num_lanes: int = 2,
                 # Legacy kwargs for backward compatibility
                 safe_distance: float = None,
                 desired_speed: float = None,
                 lane_change_gap: float = None):
        self.obs_config = obs_config or ObservationConfig()
        self.ec = expert_config or ExpertConfig()
        self.num_lanes = num_lanes

        # Legacy overrides
        if safe_distance is not None:
            self.ec.safe_distance = safe_distance
        if desired_speed is not None:
            self.ec.desired_speed = desired_speed
        if lane_change_gap is not None:
            self.ec.lane_change_gap = lane_change_gap

        # Convenience aliases
        self.safe_distance = self.ec.safe_distance
        self.desired_speed = self.ec.desired_speed
        self.lane_change_gap = self.ec.lane_change_gap

        # Episode-level bookkeeping
        self._step_count = 0
        self._last_lane_change_step = -100
        self._lane_change_cooldown = self.ec.lane_change_cooldown
        self._stuck_counter = 0

    def reset(self):
        """Reset episode-level state for a new driving episode."""
        self._step_count = 0
        self._last_lane_change_step = -100
        self._stuck_counter = 0

    def act(self, obs: np.ndarray) -> Action:
        """Parse the observation vector and decide on a driving action."""
        self._step_count += 1
        parsed = self._parse_obs(obs)
        ego_speed = parsed["ego_speed"]
        ego_lane = parsed["ego_lane"]
        neighbors = parsed["neighbors"]

        # Track consecutive stuck steps for urgency escalation
        if ego_speed < self.desired_speed - self.ec.stuck_speed_deficit:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0

        # Identify same-lane vehicles
        leader = self._find_closest_ahead(neighbors, lane_offset=0)
        second_leader = self._find_second_ahead(neighbors, lane_offset=0)
        follower = self._find_closest_behind(neighbors, lane_offset=0)

        # Longitudinal decision (now cascade-aware)
        lon = self._longitudinal_decision(ego_speed, leader, second_leader)

        # Lateral decision (now with urgency scoring and behind-awareness)
        lat = self._lateral_decision(ego_speed, ego_lane, leader,
                                     follower, neighbors)

        # Cooldown gate
        if lat != LateralAction.KEEP:
            steps_since = self._step_count - self._last_lane_change_step
            if steps_since < self._lane_change_cooldown:
                lat = LateralAction.KEEP
            else:
                self._last_lane_change_step = self._step_count

        return Action(longitudinal=lon, lateral=lat)

    # ------------------------------------------------------------------
    # Observation parsing
    # ------------------------------------------------------------------

    def _parse_obs(self, obs: np.ndarray) -> dict:
        """Decode the flat observation vector into structured data.

        Now extracts ego_accel and per-neighbor rel_accel in addition to
        the original fields.
        """
        ef = self.obs_config.ego_features
        fpn = self.obs_config.features_per_neighbor

        ego_speed = obs[0]
        ego_lane = int(round(obs[1]))
        ego_x = obs[2]
        ego_accel = obs[3] if ef >= 4 else 0.0

        k = self.obs_config.k_neighbors
        raw = obs[ef:].reshape(k, fpn)

        neighbors = []
        for i in range(k):
            if raw[i, fpn - 1] > 0.5:
                n = {
                    "rel_x": raw[i, 0],
                    "rel_lane": int(round(raw[i, 1])),
                    "rel_speed": raw[i, 2],
                }
                # rel_accel is at index 3 when fpn >= 5
                if fpn >= 5:
                    n["rel_accel"] = raw[i, 3]
                else:
                    n["rel_accel"] = 0.0
                neighbors.append(n)
        return {"ego_speed": ego_speed, "ego_lane": ego_lane,
                "ego_x": ego_x, "ego_accel": ego_accel,
                "neighbors": neighbors}

    # ------------------------------------------------------------------
    # Neighbor lookups
    # ------------------------------------------------------------------

    def _find_closest_ahead(self, neighbors, lane_offset=0):
        """Find the nearest vehicle ahead in the given lane."""
        ahead = [n for n in neighbors
                 if n["rel_lane"] == lane_offset and n["rel_x"] > 0]
        if not ahead:
            return None
        return min(ahead, key=lambda n: n["rel_x"])

    def _find_second_ahead(self, neighbors, lane_offset=0):
        """Find the second-nearest vehicle ahead in the given lane.

        Used for cascade braking detection: if the 2nd leader is braking,
        the 1st leader will soon brake too.
        """
        ahead = sorted(
            [n for n in neighbors
             if n["rel_lane"] == lane_offset and n["rel_x"] > 0],
            key=lambda n: n["rel_x"])
        if len(ahead) >= 2:
            return ahead[1]
        return None

    def _find_closest_behind(self, neighbors, lane_offset=0):
        """Find the nearest vehicle behind in the given lane."""
        behind = [n for n in neighbors
                  if n["rel_lane"] == lane_offset and n["rel_x"] < 0]
        if not behind:
            return None
        return max(behind, key=lambda n: n["rel_x"])

    # ------------------------------------------------------------------
    # Longitudinal decision
    # ------------------------------------------------------------------

    def _longitudinal_decision(self, ego_speed, leader,
                               second_leader=None) -> LongitudinalAction:
        """Decide whether to accelerate, decelerate, or maintain speed.

        Improvements over basic IDM:
        1. Cascade braking: if the 2nd leader is braking hard, preemptively
           decelerate even if the 1st leader hasn't reacted yet.
        2. Stable gap with leader pulling away: if gap < safe_distance but
           leader is accelerating away (rel_accel > 0), allow acceleration
           instead of forcing KEEP.
        """
        ec = self.ec

        if leader is not None:
            gap = leader["rel_x"]
            closing_speed = max(-leader["rel_speed"], 0.0)
            stopping_dist = (closing_speed ** 2) / (2 * ec.max_decel) + ec.stopping_buffer
            effective_safe = max(self.safe_distance, stopping_dist)

            # Cascade braking: 2nd leader decelerating hard means 1st
            # leader will brake soon. Act preemptively.
            cascade_threat = False
            if second_leader is not None:
                gap_between = second_leader["rel_x"] - gap
                if (gap_between < ec.cascade_max_gap
                        and second_leader.get("rel_accel", 0.0) < ec.cascade_accel_threshold):
                    # 2nd leader is braking hard — increase effective safe
                    # distance to give more reaction time
                    cascade_threat = True
                    effective_safe = max(effective_safe, self.safe_distance * 1.5)

            # State 1 - CLOSING: gap below safe envelope and closing fast
            if gap < effective_safe and (leader["rel_speed"] < -ec.match_speed_deadband or cascade_threat):
                return LongitudinalAction.DECELERATE

            # State 2 - STABLE GAP: gap < safe_distance, not closing fast
            if gap < self.safe_distance and leader["rel_speed"] >= -ec.match_speed_deadband:
                # Critical gap: still closing even slowly — brake to prevent
                # slow-drift collisions
                if gap < ec.critical_gap and closing_speed > 0:
                    return LongitudinalAction.DECELERATE
                # If leader is accelerating away (gap opening fast),
                # allow ego to accelerate too instead of being stuck
                leader_accel = leader.get("rel_accel", 0.0)
                if leader_accel > 1.0 and ego_speed < self.desired_speed - ec.speed_deadband_low:
                    return LongitudinalAction.ACCELERATE
                return LongitudinalAction.KEEP

        # Cruise control
        if ego_speed < self.desired_speed - ec.speed_deadband_low:
            return LongitudinalAction.ACCELERATE
        elif ego_speed > self.desired_speed + ec.speed_deadband_high:
            return LongitudinalAction.DECELERATE
        else:
            return LongitudinalAction.KEEP

    # ------------------------------------------------------------------
    # Lateral decision
    # ------------------------------------------------------------------

    def _lateral_decision(self, ego_speed, ego_lane, leader,
                          follower, neighbors) -> LateralAction:
        """Decide whether to change lanes using continuous urgency scoring.

        Replaces the binary should_pass gate with a continuous urgency
        score that considers:
        - Closing speed to leader (leader blocking)
        - Speed deficit (stuck behind slow traffic)
        - Leader's deceleration (anticipatory)
        - Tailgater pressure from behind

        Higher urgency lowers the min_improvement threshold, making lane
        changes easier to justify when the situation is more pressing.
        """
        ec = self.ec

        if leader is None:
            # Check if a fast tailgater is approaching — might want to
            # move over to let them pass, but only if urgency is high
            if follower is not None:
                closing_behind = max(follower["rel_speed"], 0.0)
                if closing_behind > 5.0 and abs(follower["rel_x"]) < 15.0:
                    # Fast approach from behind — consider moving over
                    pass  # handled by urgency below
                else:
                    return LateralAction.KEEP
            else:
                return LateralAction.KEEP

        # --- Compute continuous urgency ---
        urgency = 0.0
        leader_range = self.safe_distance * ec.leader_range_factor

        # Proactive component: closing on leader — will need to brake soon
        if leader is not None and leader["rel_speed"] < 0:
            closing = -leader["rel_speed"]
            time_to_safe = (leader["rel_x"] - self.safe_distance) / max(closing, 0.1)
            if time_to_safe < ec.proactive_lane_change_time:
                urgency += (ec.proactive_lane_change_time - time_to_safe) / ec.proactive_lane_change_time

        if leader is not None and leader["rel_x"] < leader_range:
            # Blocking component: closing on leader
            if leader["rel_speed"] < -ec.closing_speed_threshold:
                urgency += (-leader["rel_speed"] - ec.closing_speed_threshold) / 10.0

            # Stuck component: ego well below desired speed
            if ego_speed < self.desired_speed - ec.stuck_speed_deficit:
                urgency += (self.desired_speed - ec.stuck_speed_deficit - ego_speed) / 10.0

            # Anticipatory component: leader is decelerating
            leader_accel = leader.get("rel_accel", 0.0)
            if leader_accel < 0:
                urgency += abs(leader_accel) / 5.0

        # Tailgater pressure: fast car closing from behind
        if follower is not None:
            closing_behind = max(follower["rel_speed"], 0.0)
            behind_gap = abs(follower["rel_x"])
            if closing_behind > 3.0 and behind_gap < 20.0:
                urgency += closing_behind / 20.0

        if urgency < 0.1:
            return LateralAction.KEEP

        # --- Determine emergency level ---
        safe_following = max(5.0, ego_speed * 0.5)
        stuck_behind = (leader is not None
                        and leader["rel_x"] < leader_range
                        and ego_speed < self.desired_speed - ec.stuck_speed_deficit)
        emergency = (leader is not None and
                     (leader["rel_x"] < safe_following or
                      (stuck_behind and leader["rel_x"] < self.safe_distance)))

        # --- Urgency escalation from being stuck many steps ---
        urgency_escalation = min(
            self._stuck_counter / ec.urgency_escalation_steps,
            ec.max_urgency_escalation)

        # --- Evaluate candidate lanes ---
        import random
        if leader is not None:
            raw_current = leader["rel_x"]
            current_projected = raw_current + leader["rel_speed"] * ec.gap_projection_time
            current_gap = max(current_projected, 0.0)
        else:
            current_gap = float('inf')
        min_improvement = ec.min_improvement

        directions = [(-1, LateralAction.LEFT), (1, LateralAction.RIGHT)]
        safe_options = []
        for direction, lat_action in directions:
            target_lane = ego_lane + direction
            if target_lane < 0 or target_lane >= self.num_lanes:
                continue
            if self._lane_is_safe(neighbors, direction, emergency=emergency,
                                   urgency_escalation=urgency_escalation):
                ahead = self._find_closest_ahead(neighbors, direction)
                if ahead is not None:
                    raw_gap = ahead["rel_x"]
                    # Project gap forward to penalize fast-closing targets
                    projected_gap = raw_gap + ahead["rel_speed"] * ec.gap_projection_time
                    gap = max(projected_gap, 0.0)
                else:
                    gap = float('inf')

                # Check if the target lane is flowing well
                second = self._find_second_ahead(neighbors, direction)
                lane_flowing = True
                if (second is not None and ahead is not None
                        and second["rel_x"] - ahead["rel_x"] < ec.cascade_max_gap
                        and second.get("rel_accel", 0.0) < ec.cascade_accel_threshold):
                    lane_flowing = False  # target lane is congesting


                # Two-step lookahead: if the lane beyond the target is
                # much better, credit that to this intermediate move.
                # This lets the expert use L1 as a stepping stone to L0.
                second_lane = target_lane + direction
                if 0 <= second_lane < self.num_lanes:
                    far_ahead = self._find_closest_ahead(neighbors, direction * 2)
                    if far_ahead is not None:
                        far_proj = far_ahead["rel_x"] + far_ahead["rel_speed"] * ec.gap_projection_time
                        far_gap = max(far_proj, 0.0)
                    else:
                        far_gap = float('inf')
                    # Blend: effective = target_gap + factor * max(far_gap - target_gap, 0)
                    # Guard: inf - inf = nan which poisons comparisons
                    if gap < float('inf') and far_gap > gap:
                        bonus = (far_gap - gap) * ec.two_step_lookahead_factor
                        gap += bonus

                if gap > current_gap + min_improvement and lane_flowing:
                    safe_options.append((gap, lat_action))

        # --- Select among candidates ---
        if not safe_options:
            return LateralAction.KEEP
        if len(safe_options) == 1:
            return safe_options[0][1]

        # Both directions viable — prefer the one with more room
        # Guard: inf - inf = nan; treat two infinite gaps as equal
        gap_a, gap_b = safe_options[0][0], safe_options[1][0]
        if (gap_a != float('inf') or gap_b != float('inf')) and abs(gap_a - gap_b) > 5.0:
            return max(safe_options, key=lambda x: x[0])[1]
        return random.choice(safe_options)[1]

    # ------------------------------------------------------------------
    # Lane safety check
    # ------------------------------------------------------------------

    def _lane_is_safe(self, neighbors, lane_offset, emergency=False,
                      urgency_escalation=0.0) -> bool:
        """Check if the target lane has sufficient gaps for merging.

        Uses physics-based stopping distance for the ahead gap when closing
        on a target-lane vehicle, and accounts for the target-lane vehicle's
        acceleration to avoid merging into a decelerating gap.

        urgency_escalation (0..max_urgency_escalation) relaxes gap thresholds
        when the ego has been stuck behind slow traffic for many steps.
        """
        ec = self.ec
        if emergency:
            min_gap_ahead = ec.emergency_gap_ahead
            min_gap_behind = ec.emergency_gap_behind
        else:
            min_gap_ahead = max(self.lane_change_gap, ec.normal_min_gap_ahead)
            min_gap_behind = ec.normal_min_gap_behind

        # Relax ahead threshold when stuck for many consecutive steps.
        # Behind gap is a safety constraint (can't control the follower),
        # so it is never reduced by urgency.
        if urgency_escalation > 0:
            reduction = 1.0 - urgency_escalation
            min_gap_ahead *= reduction

        ahead = self._find_closest_ahead(neighbors, lane_offset)
        behind = self._find_closest_behind(neighbors, lane_offset)

        if ahead is not None:
            closing_speed = max(-ahead["rel_speed"], 0.0)
            stopping_dist = (closing_speed ** 2) / (2 * ec.max_decel) + ec.stopping_buffer
            effective_ahead = max(min_gap_ahead, stopping_dist)

            # If the target-lane vehicle ahead is decelerating, the gap
            # will close faster than current closing speed suggests —
            # add extra margin
            ahead_accel = ahead.get("rel_accel", 0.0)
            if ahead_accel < -1.0:
                effective_ahead += abs(ahead_accel) * 2.0

            if ahead["rel_x"] < effective_ahead:
                return False

        if behind is not None and abs(behind["rel_x"]) < min_gap_behind:
            return False
        return True
