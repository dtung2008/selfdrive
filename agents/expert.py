"""Rule-based expert driver for generating imitation-learning training data.

This module implements a deterministic, hand-crafted driving policy that
mimics reasonable human driving behaviour on a multi-lane highway.  It is
used to collect (observation, action) pairs that a neural-network policy
can later learn to reproduce via behavioural cloning.

High-level strategy
-------------------
1. **Longitudinal control** -- Maintain a safe following distance behind the
   lead vehicle using logic inspired by the Intelligent Driver Model (IDM).
   Accelerate toward a desired cruise speed when the road ahead is clear;
   decelerate when the gap to the leader becomes unsafe.
2. **Lane-change decisions** -- Attempt to overtake a slow leader by
   switching to an adjacent lane when there is a sufficient gap both ahead
   and behind in the target lane.  An *emergency* mode relaxes the gap
   thresholds when the ego vehicle is critically close to the leader.
3. **Lane preference** -- When neither passing nor emergency conditions
   apply, the agent holds its current lane (no gratuitous weaving).

A cooldown mechanism prevents rapid back-and-forth lane oscillations.
"""

import numpy as np
from agents.base import Agent
from utils.types import Action, LongitudinalAction, LateralAction, Observation
from utils.config import ObservationConfig


class ExpertAgent(Agent):
    """Deterministic rule-based expert driver.

    The agent reads a flat observation vector produced by the simulator,
    parses it into ego state and neighbor information, then applies a
    hierarchy of simple rules to decide on a longitudinal and lateral
    action each time step.

    Attributes:
        obs_config: Defines the observation layout (number of neighbours, etc.).
        safe_distance: Minimum acceptable gap (metres) to the lead vehicle
            before the agent begins to decelerate.
        desired_speed: Target cruise speed (m/s) on an open road.
        lane_change_gap: Minimum forward gap (metres) required in an adjacent
            lane before a lane change is considered safe.
        num_lanes: Total number of lanes on the highway (used to clamp
            lane-change candidates to valid indices).
    """

    def __init__(self, obs_config: ObservationConfig = None,
                 safe_distance: float = 20.0,
                 desired_speed: float = 30.0,
                 lane_change_gap: float = 15.0,
                 num_lanes: int = 2):
        """Initialise the expert agent with driving parameters.

        Args:
            obs_config: Observation layout descriptor.  If ``None``, the
                default :class:`ObservationConfig` is used.
            safe_distance: Desired gap (m) behind a lead vehicle.
            desired_speed: Cruise-speed target (m/s).
            lane_change_gap: Minimum forward gap (m) in the target lane to
                consider a lane change safe under normal conditions.
            num_lanes: Number of lanes on the road.
        """
        self.obs_config = obs_config or ObservationConfig()
        self.safe_distance = safe_distance
        self.desired_speed = desired_speed
        self.lane_change_gap = lane_change_gap
        self.num_lanes = num_lanes

        # Episode-level bookkeeping
        self._step_count = 0
        # Initialised to -100 so that the very first lane change is never
        # blocked by the cooldown check (step_count starts at 0).
        self._last_lane_change_step = -100
        # Minimum number of steps between consecutive lane changes to
        # prevent rapid oscillation.
        self._lane_change_cooldown = 10

    def reset(self):
        """Reset episode-level state for a new driving episode.

        Re-initialises the step counter and lane-change cooldown tracker so
        that stale state from a previous episode does not affect decisions.
        """
        self._step_count = 0
        # -100 ensures the cooldown window is already expired at step 0.
        self._last_lane_change_step = -100

    def act(self, obs: np.ndarray) -> Action:
        """Parse the observation vector and decide on a driving action.

        The decision pipeline is:
        1. Parse the flat observation into structured ego + neighbour data.
        2. Identify the lead vehicle in the current lane.
        3. Compute a longitudinal action (accel / decel / keep).
        4. Compute a lateral action (lane-left / lane-right / keep).
        5. Apply a cooldown filter to suppress rapid lane oscillation.

        Args:
            obs: Flat 1-D observation array from the simulator.

        Returns:
            An :class:`Action` with longitudinal and lateral components.
        """
        self._step_count += 1
        parsed = self._parse_obs(obs)
        ego_speed = parsed["ego_speed"]
        ego_lane = parsed["ego_lane"]
        # Each neighbour dict has keys: rel_x, rel_lane, rel_speed
        neighbors = parsed["neighbors"]

        # --- Step 2: identify the closest vehicle ahead in the same lane ---
        leader = self._find_closest_ahead(neighbors, lane_offset=0)

        # --- Step 3: longitudinal (speed) decision ---
        lon = self._longitudinal_decision(ego_speed, leader)

        # --- Step 4: lateral (lane-change) decision ---
        lat = self._lateral_decision(ego_speed, ego_lane, leader, neighbors)

        # --- Step 5: cooldown gate ---
        # If a lane change was chosen, suppress it when we changed lanes too
        # recently.  This prevents the agent from ping-ponging between lanes
        # on successive time steps.
        if lat != LateralAction.KEEP:
            steps_since = self._step_count - self._last_lane_change_step
            if steps_since < self._lane_change_cooldown:
                lat = LateralAction.KEEP
            else:
                # Commit to the lane change and record the timestamp.
                self._last_lane_change_step = self._step_count

        return Action(longitudinal=lon, lateral=lat)

    def _parse_obs(self, obs: np.ndarray) -> dict:
        """Decode the flat observation vector into structured ego and neighbour data.

        The observation layout (defined by ``obs_config``) is::

            [ego_speed, ego_lane, ego_x,
             n0_rel_x, n0_rel_lane, n0_rel_speed, n0_exists,
             n1_rel_x, n1_rel_lane, n1_rel_speed, n1_exists,
             ...]

        Args:
            obs: Flat 1-D array from the simulator.

        Returns:
            A dict with keys ``ego_speed`` (float), ``ego_lane`` (int),
            ``ego_x`` (float), and ``neighbors`` (list of dicts, each with
            ``rel_x``, ``rel_lane``, ``rel_speed``).  Only neighbours whose
            *exists* flag is set are included.
        """
        ego_speed = obs[0]
        # Lane index may be a float due to interpolation; snap to nearest int.
        ego_lane = int(round(obs[1]))
        ego_x = obs[2]

        k = self.obs_config.k_neighbors
        # Reshape the trailing portion into (k, 4) -- one row per neighbour
        # slot.  Columns: [rel_x, rel_lane, rel_speed, exists_flag].
        raw = obs[3:].reshape(k, 4)

        neighbors = []
        for i in range(k):
            # The exists flag is 1.0 when the slot is occupied and 0.0 when
            # it is padding.  Using 0.5 as a threshold for robustness.
            if raw[i, 3] > 0.5:
                neighbors.append({
                    "rel_x": raw[i, 0],
                    "rel_lane": int(round(raw[i, 1])),
                    "rel_speed": raw[i, 2],
                })
        return {"ego_speed": ego_speed, "ego_lane": ego_lane,
                "ego_x": ego_x, "neighbors": neighbors}

    def _find_closest_ahead(self, neighbors, lane_offset=0):
        """Find the nearest vehicle ahead of the ego car in a given lane.

        ``lane_offset`` is relative to the ego vehicle's current lane:
        0 = same lane, -1 = one lane to the left, +1 = one lane to the right.

        Args:
            neighbors: List of neighbour dicts (from :meth:`_parse_obs`).
            lane_offset: Relative lane index to search in.

        Returns:
            The neighbour dict with the smallest positive ``rel_x`` in the
            target lane, or ``None`` if no such vehicle exists.
        """
        # rel_x > 0 means the neighbour is ahead of the ego vehicle.
        ahead = [n for n in neighbors
                 if n["rel_lane"] == lane_offset and n["rel_x"] > 0]
        if not ahead:
            return None
        return min(ahead, key=lambda n: n["rel_x"])

    def _find_closest_behind(self, neighbors, lane_offset=0):
        """Find the nearest vehicle behind the ego car in a given lane.

        Args:
            neighbors: List of neighbour dicts (from :meth:`_parse_obs`).
            lane_offset: Relative lane index to search in.

        Returns:
            The neighbour dict with the largest (least negative) ``rel_x``
            in the target lane, or ``None`` if no such vehicle exists.
        """
        # rel_x < 0 means the neighbour is behind the ego vehicle.
        behind = [n for n in neighbors
                  if n["rel_lane"] == lane_offset and n["rel_x"] < 0]
        if not behind:
            return None
        # max() picks the value closest to zero (i.e. the nearest follower).
        return max(behind, key=lambda n: n["rel_x"])

    def _longitudinal_decision(self, ego_speed, leader) -> LongitudinalAction:
        """Decide whether to accelerate, decelerate, or maintain speed.

        The logic follows a simplified Intelligent Driver Model (IDM) with
        physics-based stopping distance:

        * If a leader exists and the gap is below the *effective* safe
          distance (which accounts for closing speed), decelerate. The
          effective distance is ``max(safe_distance, stopping_distance)``
          where ``stopping_distance = v_closing² / (2 * max_decel) + 5m``.
        * If a leader exists within ``safe_distance`` but is not approaching,
          hold speed (gap is small but stable).
        * Otherwise, accelerate toward ``desired_speed`` with a small
          deadband ([-0.5, +2.0] m/s) to avoid chattering.

        Args:
            ego_speed: Current speed of the ego vehicle (m/s).
            leader: Closest-ahead neighbour dict, or ``None`` if the lane
                is clear.

        Returns:
            A :class:`LongitudinalAction` enum value.
        """
        if leader is not None:
            gap = leader["rel_x"]
            # Physics-based safe distance: when closing on a slower leader,
            # account for the kinematic stopping distance needed to match
            # speeds. Two components (take the max):
            #   1. Base safe_distance (20m default)
            #   2. Stopping distance: v_closing² / (2 * max_decel) + buffer
            # The stopping distance term prevents late-braking collisions
            # when approaching a much slower vehicle at high speed (e.g.,
            # ego at 30 m/s, leader at 17 m/s → need ~22m to stop safely).
            closing_speed = max(-leader["rel_speed"], 0.0)
            max_decel = 5.0  # VehicleConfig.max_deceleration
            # Buffer of 10m covers vehicle length (4.5m), discrete timestep
            # quantization error (~0.6m), and safety margin.
            stopping_dist = (closing_speed ** 2) / (2 * max_decel) + 10.0
            effective_safe = max(self.safe_distance, stopping_dist)

            if gap < effective_safe and leader["rel_speed"] < 0:
                # Closing and within the safety envelope -- brake.
                return LongitudinalAction.DECELERATE
            if gap < self.safe_distance and leader["rel_speed"] >= 0:
                # Gap is small but not closing -- coast to keep it stable.
                return LongitudinalAction.KEEP

        if ego_speed < self.desired_speed - 0.5:
            # Open road and below cruise speed -- speed up.
            return LongitudinalAction.ACCELERATE
        elif ego_speed > self.desired_speed + 2.0:
            # Slightly above desired speed (asymmetric deadband accounts for
            # the fact that overshooting is less critical than undershooting).
            return LongitudinalAction.DECELERATE
        else:
            # Within the deadband -- maintain current speed.
            return LongitudinalAction.KEEP

    def _lateral_decision(self, ego_speed, ego_lane, leader,
                          neighbors) -> LateralAction:
        """Decide whether to change lanes (and which direction).

        The method first checks whether there is a *reason* to change lanes
        (a slow or blocking leader), then evaluates whether either adjacent
        lane offers a safe and meaningfully better gap.

        Two independent triggers can motivate a lane change:

        * **leader_blocking** -- the lead vehicle is within 1.5x the safe
          distance and closing at > 2 m/s.  This is the "active approach"
          trigger.
        * **stuck_behind** -- the ego speed has dropped more than 5 m/s
          below ``desired_speed`` while a leader is within 1.5x safe
          distance.  This catches the case where the ego has already matched
          a slow leader's speed and the relative speed is near zero.

        An *emergency* flag is raised when the gap is critically small
        (< 0.5 * ego_speed) or the ego is stuck with the leader inside the
        safe distance.  Emergency mode relaxes the gap requirements in the
        target lane so the agent can escape a dangerous situation.

        Args:
            ego_speed: Current ego speed (m/s).
            ego_lane: Current ego lane index.
            leader: Closest-ahead neighbour dict (or ``None``).
            neighbors: Full list of neighbour dicts.

        Returns:
            A :class:`LateralAction` -- LEFT, RIGHT, or KEEP.
        """
        # --- Determine motivation to change lanes ---
        leader_blocking = (leader is not None
                           and leader["rel_x"] < self.safe_distance * 1.5
                           and leader["rel_speed"] < -2.0)
        stuck_behind = (leader is not None
                        and leader["rel_x"] < self.safe_distance * 1.5
                        and ego_speed < self.desired_speed - 5.0)
        should_pass = leader_blocking or stuck_behind

        if not should_pass:
            # No motivation to leave the current lane.
            return LateralAction.KEEP

        # --- Determine emergency level ---
        # Speed-proportional safe following distance (clamped to 5 m at low
        # speeds to avoid near-zero thresholds).
        safe_following = ego_speed * 0.5 if ego_speed > 10 else 5.0
        emergency = (leader is not None and
                     (leader["rel_x"] < safe_following or
                      (stuck_behind and leader["rel_x"] < self.safe_distance)))

        # --- Evaluate candidate lanes ---
        import random
        current_gap = leader["rel_x"] if leader else float('inf')
        # Require a meaningful improvement so that the agent does not change
        # lanes for marginal benefit (e.g. 22 m vs. 20 m ahead).
        min_improvement = 10.0

        directions = [(-1, LateralAction.LEFT), (1, LateralAction.RIGHT)]
        safe_options = []  # list of (forward_gap, LateralAction)
        for direction, lat_action in directions:
            target_lane = ego_lane + direction
            # Clamp to valid lane indices.
            if target_lane < 0 or target_lane >= self.num_lanes:
                continue
            if self._lane_is_safe(neighbors, direction, emergency=emergency):
                # Measure the forward gap in the candidate lane.
                ahead = self._find_closest_ahead(neighbors, direction)
                gap = ahead["rel_x"] if ahead else float('inf')
                # Only worth changing if the target lane is *substantially*
                # better than the current lane.
                if gap > current_gap + min_improvement:
                    safe_options.append((gap, lat_action))

        # --- Select among candidates ---
        if not safe_options:
            return LateralAction.KEEP
        if len(safe_options) == 1:
            return safe_options[0][1]

        # Both directions are viable -- prefer the one with more room.
        if abs(safe_options[0][0] - safe_options[1][0]) > 5.0:
            return max(safe_options, key=lambda x: x[0])[1]
        # Gaps are roughly equal -- break the tie randomly to avoid a
        # systematic left/right bias.
        return random.choice(safe_options)[1]

    def _lane_is_safe(self, neighbors, lane_offset, emergency=False) -> bool:
        """Determine whether there is a sufficient gap in an adjacent lane.

        Two gap checks are performed -- one for the nearest vehicle *ahead*
        and one for the nearest vehicle *behind* in the target lane.  Both
        must exceed their respective thresholds for the lane to be deemed
        safe.

        Operating modes:
            * **Normal** (``emergency=False``): conservative thresholds
              (ahead >= 20 m, behind >= 15 m) to ensure comfortable merging.
            * **Emergency** (``emergency=True``): relaxed thresholds
              (ahead >= 10 m, behind >= 5 m) to allow escape from a
              dangerously small gap in the current lane.

        Args:
            neighbors: List of neighbour dicts.
            lane_offset: Relative lane to check (-1 for left, +1 for right).
            emergency: If ``True``, apply relaxed gap thresholds.

        Returns:
            ``True`` if both the forward and rearward gaps meet the
            required minimums; ``False`` otherwise.
        """
        if emergency:
            min_gap_ahead = 10.0
            min_gap_behind = 5.0
        else:
            # Use at least 20 m even if lane_change_gap was set lower,
            # to guarantee a safe merge under normal conditions.
            min_gap_ahead = max(self.lane_change_gap, 20.0)
            min_gap_behind = 15.0

        ahead = self._find_closest_ahead(neighbors, lane_offset)
        behind = self._find_closest_behind(neighbors, lane_offset)

        # Reject if there is a vehicle ahead in the target lane that is
        # too close. Use physics-based stopping distance when closing on
        # the target-lane NPC to avoid merging into a shrinking gap.
        if ahead is not None:
            closing_speed = max(-ahead["rel_speed"], 0.0)
            max_decel = 5.0  # VehicleConfig.max_deceleration
            stopping_dist = (closing_speed ** 2) / (2 * max_decel) + 10.0
            effective_ahead = max(min_gap_ahead, stopping_dist)
            if ahead["rel_x"] < effective_ahead:
                return False
        # Reject if there is a vehicle behind in the target lane that is
        # too close.  abs() is used because rel_x is negative for followers.
        if behind is not None and abs(behind["rel_x"]) < min_gap_behind:
            return False
        return True
