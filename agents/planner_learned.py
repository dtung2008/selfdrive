"""MPC-style planner using a LEARNED world model for rollouts.

This module implements a hybrid Model Predictive Control (MPC) planner that
combines exact ego vehicle kinematics with a learned attention-based world
model for NPC (non-player character) prediction.

Hybrid approach:
  - Ego state: exact kinematics (known physics, zero error). The ego
    vehicle's speed, position, and lane are propagated analytically using
    the known acceleration/braking parameters.
  - NPC state: an AttentionWorldModel predicts how NPC vehicles move.
    Because the WM was trained on relative observations, its predictions
    encode both NPC and ego movement. The planner decomposes the predicted
    relative deltas to extract absolute NPC motion.
  - Gap/reward: computed from exact ego positions + predicted NPC positions,
    ensuring that ego state is always accurate even if NPC predictions have
    error.

The WM predicts next-step relative observations. To recover absolute NPC
deltas:
    delta_rel_x = next_rel_x - cur_rel_x
                = (npc_x' - ego_x') - (npc_x - ego_x)
                = npc_delta_x - ego_delta_x
    => npc_delta_x = delta_rel_x + ego_delta_x

This decomposition is critical: it separates the known ego contribution
from the uncertain NPC contribution, preventing ego kinematic errors from
contaminating NPC state estimates.

Additional safety layers include:
  - A lane-change cooldown to prevent oscillating between lanes
  - A rule-based safety override using the real observation (not WM
    predictions) to catch cases where the WM hallucinates safe gaps
"""
import numpy as np
import torch
from agents.base import Agent
from models.world_model import AttentionWorldModel
from utils.types import Action, LongitudinalAction, LateralAction
from utils.config import ModelConfig, SimConfig, VehicleConfig, RoadConfig, ObservationConfig


class PlannerLearnedModel(Agent):
    """Hybrid MPC planner: exact ego dynamics + learned NPC prediction.

    This planner uses random-shooting MPC: it generates many candidate
    action sequences, simulates each one forward using exact ego kinematics
    and the learned world model for NPC prediction, scores each rollout
    with a reward function, and returns the first action of the best
    sequence.

    Unlike PlannerTrueModel, this planner cannot access the real simulator
    for rollouts. Instead it relies on the AttentionWorldModel, which
    introduces prediction error that compounds over the horizon. To mitigate
    this, the planning horizon is kept short (typically 4 steps vs. the
    true planner's longer horizon).

    Attributes:
        wm: The trained AttentionWorldModel for NPC state prediction.
        mc: ModelConfig with planner parameters (horizon, rollouts, discount).
        sc: SimConfig with simulation parameters (dt timestep).
        vc: VehicleConfig with vehicle dynamics (max accel/decel, speed limits).
        rc: RoadConfig with road parameters (num_lanes, speed_limit).
        normalizer: Optional observation normalizer for WM input/output.
        safety_margin: Extra margin (meters) for safety checks.
        rng: Seeded random number generator for reproducibility.
        device: Torch device (CPU/GPU) for WM inference.
        horizon: Planning horizon in timesteps.
        debug: If True, print detailed per-step rollout diagnostics.
        _step_count: Running count of act() calls in the current episode.
        _last_lane_change_step: Step at which the last lane change occurred.
        _lane_change_cooldown: Minimum steps required between lane changes.
    """

    def __init__(self, world_model: AttentionWorldModel,
                 model_config: ModelConfig = None,
                 sim_config: SimConfig = None,
                 vehicle_config: VehicleConfig = None,
                 road_config: RoadConfig = None,
                 obs_config: ObservationConfig = None,
                 normalizer=None,
                 safety_margin: float = 2.0,
                 seed: int = 42):
        """Initialize the learned-model planner.

        Args:
            world_model: A trained AttentionWorldModel instance.
            model_config: Configuration for planner parameters. Uses defaults
                if None.
            sim_config: Configuration for simulation timestep (dt). Uses
                defaults if None.
            vehicle_config: Configuration for vehicle dynamics (acceleration,
                speed limits). Uses defaults if None.
            road_config: Configuration for road layout (number of lanes,
                speed limit). Uses defaults if None.
            obs_config: Observation layout descriptor. If None, the default
                ObservationConfig is used.
            normalizer: Optional normalizer with transform() and
                inverse_transform() methods for WM input/output.
            safety_margin: Extra safety margin in meters (currently stored
                but safety checks use fixed thresholds).
            seed: Random seed for reproducible rollout sampling.
        """
        self.wm = world_model
        self.mc = model_config or ModelConfig()
        self.sc = sim_config or SimConfig()
        self.vc = vehicle_config or VehicleConfig()
        self.rc = road_config or RoadConfig()
        self.oc = obs_config or ObservationConfig()
        self.normalizer = normalizer
        self.safety_margin = safety_margin
        self.rng = np.random.RandomState(seed)
        self.device = next(world_model.parameters()).device
        # Use WM-specific horizon from config (default 4).
        # WM prediction errors compound over steps, so this is typically
        # much shorter than planner_true's horizon.
        self.horizon = self.mc.planner_wm_horizon
        self.debug = False
        self._step_count = 0
        self._last_lane_change_step = -100  # lane-change cooldown tracker
        self._lane_change_cooldown = 10     # min steps between lane changes

    def reset(self):
        """Reset episode state for a new episode.

        Clears the step counter and lane-change cooldown tracker so that
        the planner starts fresh without residual state from the previous
        episode.
        """
        self._step_count = 0
        self._last_lane_change_step = -100

    def _get_accel(self, action: Action) -> float:
        """Convert a discrete longitudinal action to a continuous acceleration.

        Args:
            action: The Action whose longitudinal component to convert.

        Returns:
            Acceleration in m/s^2: positive for ACCELERATE, negative for
            DECELERATE, zero for KEEP.
        """
        if action.longitudinal == LongitudinalAction.ACCELERATE:
            return self.vc.max_acceleration
        elif action.longitudinal == LongitudinalAction.DECELERATE:
            return -self.vc.max_deceleration
        return 0.0

    def _get_lane_change(self, action: Action) -> int:
        """Extract the lane-change delta from an action.

        Args:
            action: The Action whose lateral component to convert.

        Returns:
            Integer lane delta: -1 for left, 0 for keep, +1 for right.
        """
        return int(action.lateral)

    def act(self, obs: np.ndarray) -> Action:
        """Select the best action via random-shooting MPC with hybrid rollouts.

        The method performs the following steps:
          1. Generate candidate action sequences (single-action repeats +
             random sequences).
          2. For each candidate, simulate forward over the planning horizon:
             - Ego state is propagated with exact kinematics.
             - NPC states are predicted using the learned world model.
          3. Score each rollout with the reward function.
          4. Return the first action of the highest-scoring sequence, after
             applying lane-change cooldown and safety overrides.

        Args:
            obs: The current raw observation vector with layout:
                [ego_speed, ego_lane, ego_x,
                 npc0_rel_x, npc0_rel_lane, npc0_rel_speed, npc0_exists, ...]

        Returns:
            The Action to execute this timestep.
        """
        num_actions = Action.num_actions()
        horizon = self.horizon
        num_rollouts = self.mc.planner_num_rollouts
        dt = self.sc.dt
        ef = self.oc.ego_features
        fpn = self.oc.features_per_neighbor
        obs_dim = len(obs)
        k = self.oc.k_neighbors
        num_lanes = self.rc.num_lanes

        # --- Generate candidate action sequences ---
        # First `num_actions` rollouts are single-action repeats (one per
        # discrete action), providing a baseline for every primitive action.
        # Remaining rollouts are random sequences for exploration diversity.
        all_actions = np.zeros((num_rollouts, horizon), dtype=int)
        for i in range(num_rollouts):
            if i < num_actions:
                # Single-action repeat: action i held for the full horizon
                all_actions[i, :] = i
            else:
                # Random sequence for exploration
                all_actions[i, :] = self.rng.randint(0, num_actions, size=horizon)

        # --- Parse initial ego and NPC state from observation ---
        ego_speed_init = obs[0]
        ego_lane_init = int(round(obs[1]))
        ego_x_init = obs[2]

        # Convert relative NPC observations to absolute positions/speeds.
        # We track absolute NPC state so that we can recompute relative
        # positions from exact ego state at each rollout step.
        # obs format: [ego_speed, ego_lane, ego_x, rel_x, rel_lane, rel_speed, exists, ...]
        npc_abs_x = np.zeros((num_rollouts, k))
        npc_abs_speed = np.zeros((num_rollouts, k))
        npc_lane = np.zeros((num_rollouts, k))
        npc_exists = np.zeros((num_rollouts, k))
        for ni in range(k):
            base = ef + ni * fpn
            rel_x = obs[base]
            rel_lane = obs[base + 1]
            rel_speed = obs[base + 2]
            exists = obs[base + fpn - 1]
            # Broadcast initial NPC state across all rollouts (all start
            # from the same observation)
            npc_abs_x[:, ni] = ego_x_init + rel_x
            npc_abs_speed[:, ni] = ego_speed_init + rel_speed
            npc_lane[:, ni] = ego_lane_init + rel_lane
            npc_exists[:, ni] = exists

        # Initialize per-rollout ego state arrays (all start identical)
        ego_speeds = np.full(num_rollouts, ego_speed_init)
        ego_xs = np.full(num_rollouts, ego_x_init)
        ego_lanes = np.full(num_rollouts, ego_lane_init, dtype=int)

        total_rewards = np.zeros(num_rollouts)
        discount = 1.0

        # Debug arrays: track per-step rewards and ego state for the first
        # `num_actions` rollouts (the single-action repeats), which are
        # easiest to interpret in diagnostic output
        if self.debug:
            debug_per_step = np.zeros((num_actions, horizon))
            debug_ego_speeds = np.zeros((num_actions, horizon))
            debug_ego_lanes = np.zeros((num_actions, horizon), dtype=int)
            debug_ego_xs = np.zeros((num_actions, horizon))

        # Set WM to eval mode (disables dropout, etc.) and disable gradient
        # tracking for inference efficiency
        self.wm.eval()
        first_effective_actions = None
        with torch.no_grad():
            for t in range(horizon):
                # Snapshot ego state before applying this step's action,
                # needed for constructing the WM input observation and
                # for computing ego deltas
                prev_speeds = ego_speeds.copy()
                prev_xs = ego_xs.copy()
                prev_lanes = ego_lanes.copy()

                # --- Step 1: Propagate ego state with exact kinematics ---
                # Also track "effective" actions: if a lane change is clipped
                # at the road boundary, the WM should see KEEP instead of
                # the clipped lane change, since the ego didn't actually
                # change lanes.
                effective_actions = all_actions[:, t].copy()
                for i in range(num_rollouts):
                    action = Action.from_index(all_actions[i, t])

                    # Apply longitudinal dynamics: v' = clip(v + a*dt)
                    accel = self._get_accel(action)
                    ego_speeds[i] = np.clip(
                        ego_speeds[i] + accel * dt,
                        self.vc.min_speed, self.vc.max_speed)
                    # Update position: x' = x + v' * dt
                    ego_xs[i] += ego_speeds[i] * dt

                    # Apply lateral dynamics: lane change clipped to road bounds
                    lane_delta = self._get_lane_change(action)
                    new_lane = np.clip(
                        ego_lanes[i] + lane_delta, 0, num_lanes - 1)

                    # If lane change was clipped (ego stayed in same lane),
                    # remap the action index to preserve the longitudinal
                    # component but set lateral to KEEP. This ensures the WM
                    # sees the action that actually happened.
                    if new_lane == ego_lanes[i] and lane_delta != 0:
                        # Action index layout: idx = lon_idx * 3 + lat_idx
                        # where lat_idx: 0=LEFT, 1=KEEP, 2=RIGHT
                        lon_idx = all_actions[i, t] // 3
                        effective_actions[i] = lon_idx * 3 + 1  # lateral KEEP=1
                    ego_lanes[i] = new_lane

                # Save the effective actions at t=0: these correspond to the
                # actual first action each rollout would execute
                if t == 0:
                    first_effective_actions = effective_actions.copy()

                # --- Step 2: Construct WM input observation ---
                # The WM expects relative observations computed from the
                # PRE-action ego state (prev_*), matching the training data
                # format where obs[t] + action -> obs[t+1].
                obs_for_wm = np.zeros((num_rollouts, obs_dim), dtype=np.float32)
                obs_for_wm[:, 0] = prev_speeds
                obs_for_wm[:, 1] = prev_lanes
                obs_for_wm[:, 2] = prev_xs
                for ni in range(k):
                    base = ef + ni * fpn
                    # Recompute relative positions from absolute NPC state
                    # and pre-action ego state
                    obs_for_wm[:, base] = npc_abs_x[:, ni] - prev_xs       # rel_x
                    obs_for_wm[:, base + 1] = npc_lane[:, ni] - prev_lanes  # rel_lane
                    obs_for_wm[:, base + 2] = npc_abs_speed[:, ni] - prev_speeds  # rel_speed
                    obs_for_wm[:, base + fpn - 1] = npc_exists[:, ni]

                # --- Step 3: Run the world model ---
                # Normalize input, run WM inference, denormalize output
                if self.normalizer is not None:
                    obs_norm = self.normalizer.transform(obs_for_wm)
                else:
                    obs_norm = obs_for_wm
                obs_t = torch.FloatTensor(obs_norm).to(self.device)
                actions_t = torch.LongTensor(effective_actions).to(self.device)

                # WM predicts the full next observation in normalized space
                pred_norm = self.wm(obs_t, actions_t)
                if self.normalizer is not None:
                    pred_real = self.normalizer.inverse_transform(
                        pred_norm.cpu().numpy())
                else:
                    pred_real = pred_norm.cpu().numpy()

                # --- Step 4: Extract absolute NPC deltas from WM output ---
                # The WM was trained on relative observations, so its
                # predictions encode both NPC and ego movement:
                #   delta_rel_x = next_rel_x - cur_rel_x
                #               = (npc_x' - ego_x') - (npc_x - ego_x)
                #               = npc_delta_x - ego_delta_x
                # Therefore: npc_delta_x = delta_rel_x + ego_delta_x
                # This decomposition lets us use exact ego deltas while
                # only relying on the WM for the NPC component.
                ego_delta_x = ego_xs - prev_xs
                ego_delta_speed = ego_speeds - prev_speeds

                for ni in range(k):
                    base = ef + ni * fpn
                    # WM-predicted next relative position and speed
                    pred_rel_x = pred_real[:, base]
                    pred_rel_speed = pred_real[:, base + 2]
                    # Current relative position and speed (WM input)
                    cur_rel_x = obs_for_wm[:, base]
                    cur_rel_speed = obs_for_wm[:, base + 2]

                    # Change in relative coordinates as predicted by WM
                    delta_rel_x = pred_rel_x - cur_rel_x
                    delta_rel_speed = pred_rel_speed - cur_rel_speed

                    # Recover absolute NPC deltas by adding back the known
                    # ego movement that was subtracted in the relative frame
                    npc_delta_x = delta_rel_x + ego_delta_x
                    npc_delta_speed = delta_rel_speed + ego_delta_speed

                    # Update absolute NPC state
                    npc_abs_x[:, ni] += npc_delta_x
                    npc_abs_speed[:, ni] += npc_delta_speed

                # --- Step 5: Compute reward for this timestep ---
                # Uses exact ego state + WM-predicted NPC state
                rewards = self._estimate_reward(
                    prev_speeds, ego_speeds, ego_xs, ego_lanes, prev_lanes,
                    npc_abs_x, npc_abs_speed, npc_lane, npc_exists,
                    effective_actions)
                total_rewards += discount * rewards
                discount *= self.mc.planner_discount

                # --- Debug logging ---
                if self.debug:
                    for i in range(num_actions):
                        debug_per_step[i, t] = rewards[i]
                        debug_ego_speeds[i, t] = ego_speeds[i]
                        debug_ego_lanes[i, t] = ego_lanes[i]
                        debug_ego_xs[i, t] = ego_xs[i]

                    # Track NPC positions for KP/- (action index 3) to
                    # diagnose cases where the WM hallucinates phantom gaps
                    if not hasattr(self, '_debug_npc_data'):
                        self._debug_npc_data = {}
                    kp_idx = 3  # KP/- is the 4th action (index 3)
                    self._debug_npc_data[t] = {
                        'ego_x': ego_xs[kp_idx],
                        'ego_lane': ego_lanes[kp_idx],
                        'npc_xs': npc_abs_x[kp_idx].copy(),
                        'npc_lanes': npc_lane[kp_idx].copy(),
                        'npc_exists': npc_exists[kp_idx].copy(),
                        'npc_speeds': npc_abs_speed[kp_idx].copy(),
                    }

        # --- Select the best rollout ---
        # Tie-breaking: among rollouts with max reward (within floating-point
        # tolerance), prefer those whose first effective action keeps the
        # current lane. This reduces unnecessary lane oscillation.
        max_reward = total_rewards.max()
        best_candidates = np.where(np.abs(total_rewards - max_reward) < 1e-6)[0]
        keep_candidates = [c for c in best_candidates
                           if Action.from_index(int(first_effective_actions[c])).lateral.value == 0]
        if keep_candidates:
            best_idx = self.rng.choice(keep_candidates)
        else:
            best_idx = self.rng.choice(best_candidates)
        self._step_count += 1

        # --- Debug output: detailed rollout diagnostics ---
        if self.debug:
            # Human-readable names for the 9 discrete actions
            # Format: longitudinal/lateral (ACC=accelerate, KP=keep, BRK=brake,
            # -=keep lane, L=left, R=right)
            action_names = ['ACC/-', 'ACC/L', 'ACC/R',
                            ' KP/-', ' KP/L', ' KP/R',
                            'BRK/-', 'BRK/L', 'BRK/R']
            print(f"\n=== Step {self._step_count} | ego_spd={ego_speed_init:.1f} lane={ego_lane_init} ===")

            # Summary table: total reward and per-step breakdown for each
            # single-action rollout
            print(f"  {'Action':>6} | {'Total':>8} | per-step rewards")
            for i in range(num_actions):
                steps_str = " ".join(f"{debug_per_step[i,t]:+.3f}" for t in range(horizon))
                marker = " <== BEST" if i == best_idx else ""
                print(f"  {action_names[i]:>6} | {total_rewards[i]:>8.3f} | [{steps_str}]{marker}")

            # Detailed comparison of lane-change actions vs KP/- (keep
            # speed, keep lane) to understand why lane changes are or
            # aren't selected
            keep_idx = 3  # KP/-
            for lc_idx in [2, 5, 8, 1, 4, 7]:  # ACC/R, KP/R, BRK/R, ACC/L, KP/L, BRK/L
                lc_name = action_names[lc_idx]
                diff = total_rewards[lc_idx] - total_rewards[keep_idx]
                # Only show comparisons where the difference is small enough
                # to be relevant for decision-making
                if abs(diff) < 5.0:
                    print(f"\n  {lc_name} vs KP/- detail (diff={diff:+.3f}):")
                    for t in range(horizon):
                        r_lc = debug_per_step[lc_idx, t]
                        r_kp = debug_per_step[keep_idx, t]
                        spd_lc = debug_ego_speeds[lc_idx, t]
                        spd_kp = debug_ego_speeds[keep_idx, t]
                        ln_lc = debug_ego_lanes[lc_idx, t]
                        ln_kp = debug_ego_lanes[keep_idx, t]
                        print(f"    t={t}: {lc_name} r={r_lc:+.3f} spd={spd_lc:.1f} ln={ln_lc}"
                              f"  |  KP/- r={r_kp:+.3f} spd={spd_kp:.1f} ln={ln_kp}"
                              f"  |  delta={r_lc-r_kp:+.3f}")

            # If the best rollout is a random sequence (not a single-action
            # repeat), report which random rollout won
            if best_idx >= num_actions:
                first_act = int(all_actions[best_idx, 0])
                print(f"\n  Best is random rollout #{best_idx}, first_act={action_names[first_act]}, "
                      f"reward={total_rewards[best_idx]:.3f}")
            print(f"  Chosen: {action_names[int(all_actions[best_idx, 0])]}")

            # Dump WM-predicted NPC positions for KP/- when it received a
            # negative reward, to help diagnose WM hallucination issues
            if hasattr(self, '_debug_npc_data') and any(debug_per_step[3, t] < 0 for t in range(horizon)):
                print(f"\n  WM NPC predictions for KP/- (diagnosing bad reward):")
                for t in range(horizon):
                    d = self._debug_npc_data[t]
                    ego_x = d['ego_x']
                    ego_ln = d['ego_lane']
                    print(f"    t={t}: ego_x={ego_x:.1f} ego_lane={ego_ln:.0f}")
                    for ni in range(len(d['npc_xs'])):
                        if d['npc_exists'][ni] > 0.5:
                            gap = d['npc_xs'][ni] - ego_x
                            same = "SAME" if abs(d['npc_lanes'][ni] - ego_ln) < 0.5 else "diff"
                            print(f"      NPC{ni}: lane={d['npc_lanes'][ni]:.0f} "
                                  f"x={d['npc_xs'][ni]:.1f} gap={gap:+.1f} "
                                  f"spd={d['npc_speeds'][ni]:.1f} [{same}]")
            # Clean up debug data to prevent accumulation across steps
            if hasattr(self, '_debug_npc_data'):
                del self._debug_npc_data

        # Convert the best rollout's first effective action to an Action object
        chosen = Action.from_index(int(first_effective_actions[best_idx]))

        # --- Lane-change cooldown ---
        # Prevent rapid lane oscillation (e.g., left-right-left) by
        # enforcing a minimum number of steps between lane changes
        if chosen.lateral.value != 0:
            steps_since = self._step_count - self._last_lane_change_step
            if steps_since < self._lane_change_cooldown:
                # Block the lane change but keep the longitudinal action
                chosen = Action(chosen.longitudinal, LateralAction.KEEP)
            else:
                # Allow the lane change and record when it happened
                self._last_lane_change_step = self._step_count

        # --- Safety override using REAL observation ---
        # Apply rule-based safety checks on the actual observation (not
        # WM predictions) to catch cases where the WM hallucinates safe
        # gaps that do not exist in reality
        chosen = self._apply_safety(obs, chosen)

        return chosen

    def _apply_safety(self, obs: np.ndarray, action: Action) -> Action:
        """Override dangerous actions based on the current real observation.

        This is a rule-based safety layer that operates on the raw observation
        (not WM predictions) to catch real-world dangers the WM might miss.

        Two operating modes:
        - Normal: Block lane changes when the target lane gap is < 15m.
        - Emergency: When the same-lane gap ahead is < 10m (imminent
          collision risk), lane changes are allowed with relaxed clearance
          thresholds (5m) to enable evasive maneuvers.

        In both modes, forced braking is applied when the gap ahead is below
        a speed-dependent safe following distance (half-second rule) or below
        an absolute 10m minimum.

        Args:
            obs: The raw (un-normalized) observation vector.
            action: The action chosen by the planning loop.

        Returns:
            The (possibly overridden) Action. Returned unchanged if safe.
        """
        ef = self.oc.ego_features
        fpn = self.oc.features_per_neighbor
        k = self.oc.k_neighbors

        ego_speed = obs[0]
        ego_lane = int(round(obs[1]))
        lateral = action.lateral.value
        target_lane = ego_lane + lateral

        # Scan all NPCs to find minimum gaps in the current and target lanes.
        # Also track the closing speed to the nearest same-lane NPC ahead,
        # which is needed for physics-based stopping distance.
        min_gap_ahead = float('inf')      # closest NPC ahead in ego's lane
        closing_speed_ahead = 0.0         # closing speed to that NPC (positive = approaching)
        min_target_ahead = float('inf')   # closest NPC ahead in target lane
        min_target_behind = float('inf')  # closest NPC behind in target lane
        closing_speed_target = 0.0        # closing speed to nearest target-lane NPC ahead

        for i in range(k):
            base = ef + i * fpn
            rel_x = obs[base]
            rel_lane = obs[base + 1]
            rel_speed = obs[base + 2]     # npc_speed - ego_speed (negative = closing)
            exists = obs[base + fpn - 1]
            if exists < 0.5:
                continue
            npc_lane = ego_lane + rel_lane

            # Same-lane ahead: track the closest vehicle and its closing speed
            if abs(rel_lane) < 0.5 and rel_x > 0:
                if rel_x < min_gap_ahead:
                    min_gap_ahead = rel_x
                    # rel_speed = npc_speed - ego_speed, so negative means
                    # ego is faster (closing). Convert to positive closing speed.
                    closing_speed_ahead = max(-rel_speed, 0.0)

            # Target lane: track gaps and closing speed for merge safety
            if lateral != 0 and abs(npc_lane - target_lane) < 0.5:
                if rel_x > 0:
                    if rel_x < min_target_ahead:
                        min_target_ahead = rel_x
                        closing_speed_target = max(-rel_speed, 0.0)
                else:
                    min_target_behind = min(min_target_behind, abs(rel_x))

        # Emergency detection: same-lane gap < 10m means we are about to
        # collide if we stay in this lane
        emergency = min_gap_ahead < 10.0

        # Block unsafe lane changes based on target lane clearance.
        # Use physics-based stopping distance for the ahead threshold when
        # closing on a target-lane NPC, so we don't merge into a shrinking gap.
        if lateral != 0:
            if emergency:
                # In emergency, allow tighter merges for evasion but still
                # require 5m minimum to avoid side collisions
                if min_target_ahead < 5.0 or min_target_behind < 5.0:
                    action = Action(action.longitudinal, LateralAction.KEEP)
            else:
                # Physics-based ahead clearance: if closing on the target-lane
                # NPC, require enough room to stop safely after merging
                target_stopping = (closing_speed_target ** 2) / (2 * self.vc.max_deceleration) + 10.0
                safe_ahead = max(15.0, target_stopping)
                if min_target_ahead < safe_ahead or min_target_behind < 15.0:
                    action = Action(action.longitudinal, LateralAction.KEEP)

        # Three-state following distance logic (same-lane only), mirroring
        # the expert agent's IDM-inspired approach:
        #
        # State 1 - CLOSING (closing_speed > 0.5 m/s): Use physics-based
        #   stopping distance to decide when to brake. The safe gap is the max
        #   of time-headway (0.7s), a 20m floor, and the kinematic stopping
        #   distance (v_closing² / 2a + buffer). Force DECELERATE if violated.
        #
        # State 2 - STABLE (not closing, gap < 20m): Gap is small but not
        #   shrinking — speeds are matched. Cap action at KEEP to prevent
        #   re-accelerating into the NPC, but don't force braking. This avoids
        #   the oscillation where forced braking overshoots, opens a huge gap,
        #   then ego re-accelerates and repeats.
        #
        # State 3 - CRITICAL (gap < 10m): Always brake regardless of closing
        #   speed — too close for comfort at any speed differential.
        max_decel = self.vc.max_deceleration  # 5.0 m/s²

        if closing_speed_ahead > 0.5:
            # Closing: physics-based safe distance
            stopping_dist = (closing_speed_ahead ** 2) / (2 * max_decel) + 10.0
            safe_gap = max(ego_speed * 0.7, 20.0, stopping_dist)
            if min_gap_ahead < safe_gap:
                if action.longitudinal != LongitudinalAction.DECELERATE:
                    action = Action(LongitudinalAction.DECELERATE, action.lateral)
        else:
            # Not closing: gap is stable or opening. Prevent acceleration
            # if gap is still under the safe threshold, but allow KEEP so
            # ego matches NPC speed without overshooting downward.
            if min_gap_ahead < 20.0:
                if action.longitudinal == LongitudinalAction.ACCELERATE:
                    action = Action(LongitudinalAction.KEEP, action.lateral)

        # Absolute minimum gap override: always brake if gap < 10m
        if min_gap_ahead < 10.0:
            action = Action(LongitudinalAction.DECELERATE, action.lateral)

        return action

    def _estimate_reward(self, prev_speeds, ego_speeds, ego_xs,
                         ego_lanes, prev_lanes,
                         npc_xs, npc_speeds, npc_lanes, npc_exists,
                         actions) -> np.ndarray:
        """Estimate per-step reward for a batch of rollouts.

        The reward function cleanly separates safety from optimization:

        1. OPTIMIZATION (base reward): Reward proportional to ego speed
           normalized by the speed limit, encouraging the agent to drive
           fast. Exceeding the speed limit incurs a linearly increasing
           penalty. A small lane-change cost breaks near-ties in favor
           of staying in lane.

        2. SAFETY (override): Catastrophic events override the base reward:
           - Collision (same-lane gap < 5m): reward = -100
           - Lane change into occupied space (gap < 15m ahead or
             < 10m behind): reward = -100
           - Too close to vehicle ahead (gap < speed * 0.5s): reward
             capped at 0 (not catastrophic, but clearly worse than
             any positive speed reward)

        Args:
            prev_speeds: Ego speeds before this step, shape (batch,).
            ego_speeds: Ego speeds after this step, shape (batch,).
            ego_xs: Ego positions after this step, shape (batch,).
            ego_lanes: Ego lanes after this step, shape (batch,).
            prev_lanes: Ego lanes before this step, shape (batch,).
            npc_xs: Absolute NPC positions, shape (batch, k).
            npc_speeds: Absolute NPC speeds, shape (batch, k).
            npc_lanes: NPC lane indices, shape (batch, k).
            npc_exists: NPC existence flags, shape (batch, k).
            actions: Effective action indices for this step, shape (batch,).

        Returns:
            A numpy array of shape (batch,) containing the reward for
            each rollout at this timestep.
        """
        batch = len(ego_speeds)
        rewards = np.zeros(batch)

        # === OPTIMIZATION: Speed reward ===
        # Reward = speed / speed_limit, capped at 1.0 when at the limit.
        # Exceeding the limit linearly reduces reward (penalty of 1/5 per
        # m/s over the limit).
        speed_limit = max(self.rc.speed_limit, 1e-6)  # avoid division by zero
        for i in range(batch):
            if ego_speeds[i] <= speed_limit:
                rewards[i] = ego_speeds[i] / speed_limit
            else:
                # Linear penalty for exceeding speed limit
                excess = ego_speeds[i] - speed_limit
                rewards[i] = 1.0 - excess / 5.0

        # Small lane-change cost to prefer KEEP when speed rewards are
        # similar. At 30 m/s the speed reward is 1.0, so 0.01 (~1%) is
        # negligible for decision-making but breaks near-ties in favor of
        # staying in the current lane, reducing unnecessary oscillation.
        changed_lane_action = (ego_lanes != prev_lanes)
        rewards[changed_lane_action] -= 0.01

        # === SAFETY CHECKS ===
        k = npc_xs.shape[1]  # number of NPC vehicles
        for n in range(k):
            # Compute gap: positive = NPC is ahead of ego
            raw_gap = npc_xs[:, n] - ego_xs
            # Boolean mask: NPC is in the same lane AND exists
            same_lane = (np.abs(npc_lanes[:, n] - ego_lanes) < 0.5) & \
                        (npc_exists[:, n] > 0.5)

            # Check 1: Same-lane collision -- gap < 8m in the same lane.
            # The actual simulator triggers collision at ~4.5m (vehicle length),
            # but 8m gives the planner enough margin to avoid committing to
            # actions that lead to unavoidable collisions within the short
            # WM planning horizon.
            collision = same_lane & (np.abs(raw_gap) < 8.0)
            rewards[collision] = -100.0

            # Check 2: Same-lane too close ahead -- three-state logic matching
            # _apply_safety. Only penalize when actually closing on the NPC;
            # stable/opening gaps at moderate distance are acceptable.
            ahead = same_lane & (raw_gap > 0)
            closing_speed = np.maximum(ego_speeds - npc_speeds[:, n], 0.0)
            closing = closing_speed > 0.5

            # 2a: Closing approach -- physics-based safe distance.
            # Penalize rollouts where ego is closing too fast on a slower NPC.
            max_decel = 5.0  # self.vc.max_deceleration
            stopping_dist = (closing_speed ** 2) / (2 * max_decel) + 10.0
            min_gap = np.maximum(np.maximum(ego_speeds * 0.7, 20.0), stopping_dist)
            too_close_closing = ahead & closing & (raw_gap < min_gap)
            rewards[too_close_closing] = np.minimum(rewards[too_close_closing], -10.0)

            # 2b: Stable/opening gap but critically close (< 10m) -- still
            # dangerous even if not actively closing.
            critically_close = ahead & ~closing & (raw_gap < 10.0)
            rewards[critically_close] = np.minimum(rewards[critically_close], -10.0)

        # Check 3: Lane change into occupied space.
        # If ego just changed lanes, check that the target lane has
        # sufficient clearance. The ahead threshold uses physics-based
        # stopping distance when closing on the target-lane NPC, so we
        # penalize merging into a shrinking gap.
        changed_lane = (ego_lanes != prev_lanes)
        if np.any(changed_lane):
            for n in range(k):
                # Skip NPCs that don't exist in any rollout (early exit)
                if not np.any(npc_exists[:, n] > 0.5):
                    continue
                # Find rollouts where this NPC is in the NEW lane and exists
                in_new_lane = (np.abs(npc_lanes[:, n] - ego_lanes) < 0.5) & \
                              (npc_exists[:, n] > 0.5) & changed_lane
                if not np.any(in_new_lane):
                    continue
                gap = npc_xs[:, n] - ego_xs  # positive = NPC ahead
                # Physics-based ahead clearance: closing_speed² / (2a) + 10m
                closing = np.maximum(ego_speeds - npc_speeds[:, n], 0.0)
                stopping_dist = (closing ** 2) / (2 * max_decel) + 10.0
                safe_ahead = np.maximum(15.0, stopping_dist)
                unsafe_ahead = in_new_lane & (gap > 0) & (gap < safe_ahead)
                unsafe_behind = in_new_lane & (gap <= 0) & (np.abs(gap) < 10.0)
                rewards[unsafe_ahead | unsafe_behind] = -100.0

        return rewards
