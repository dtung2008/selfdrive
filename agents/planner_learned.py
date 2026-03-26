"""MPC-style planner using a LEARNED world model for rollouts.

Hybrid approach:
  - Ego state: exact kinematics (known physics, zero error)
  - NPC state: attention world model predicts how NPCs move
  - Gap: computed externally from exact ego + predicted NPC positions

The WM is used to predict NPC relative-state deltas. These deltas
include ego movement effects (since the WM was trained on relative
observations), so we subtract the ego movement to get absolute NPC
movement, then recompute relative positions from exact ego state.
"""
import numpy as np
import torch
from agents.base import Agent
from models.world_model import AttentionWorldModel
from utils.types import Action, LongitudinalAction, LateralAction
from utils.config import ModelConfig, SimConfig, VehicleConfig, RoadConfig


class PlannerLearnedModel(Agent):
    """Hybrid planner: exact ego dynamics + learned NPC prediction."""

    def __init__(self, world_model: AttentionWorldModel,
                 model_config: ModelConfig = None,
                 sim_config: SimConfig = None,
                 vehicle_config: VehicleConfig = None,
                 road_config: RoadConfig = None,
                 normalizer=None,
                 safety_margin: float = 2.0,
                 seed: int = 42):
        self.wm = world_model
        self.mc = model_config or ModelConfig()
        self.sc = sim_config or SimConfig()
        self.vc = vehicle_config or VehicleConfig()
        self.rc = road_config or RoadConfig()
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
        """Reset episode state."""
        self._step_count = 0
        self._last_lane_change_step = -100

    def _get_accel(self, action: Action) -> float:
        if action.longitudinal == LongitudinalAction.ACCELERATE:
            return self.vc.max_acceleration
        elif action.longitudinal == LongitudinalAction.DECELERATE:
            return -self.vc.max_deceleration
        return 0.0

    def _get_lane_change(self, action: Action) -> int:
        """Return lane delta: -1 left, 0 keep, +1 right."""
        return int(action.lateral)

    def act(self, obs: np.ndarray) -> Action:
        """Run random-shooting MPC with hybrid rollouts."""
        num_actions = Action.num_actions()
        horizon = self.horizon
        num_rollouts = self.mc.planner_num_rollouts
        dt = self.sc.dt
        obs_dim = len(obs)
        k = (obs_dim - 3) // 4
        num_lanes = self.rc.num_lanes

        # Generate action sequences with action-repeat bias
        all_actions = np.zeros((num_rollouts, horizon), dtype=int)
        for i in range(num_rollouts):
            if i < num_actions:
                all_actions[i, :] = i
            else:
                all_actions[i, :] = self.rng.randint(0, num_actions, size=horizon)

        # Parse initial state
        ego_speed_init = obs[0]
        ego_lane_init = int(round(obs[1]))
        ego_x_init = obs[2]

        # Extract initial absolute NPC positions from relative obs
        # obs format: [ego_speed, ego_lane, ego_x, rel_x, rel_lane, rel_speed, exists, ...]
        npc_abs_x = np.zeros((num_rollouts, k))
        npc_abs_speed = np.zeros((num_rollouts, k))
        npc_lane = np.zeros((num_rollouts, k))
        npc_exists = np.zeros((num_rollouts, k))
        for ni in range(k):
            base = 3 + ni * 4
            rel_x = obs[base]
            rel_lane = obs[base + 1]
            rel_speed = obs[base + 2]
            exists = obs[base + 3]
            npc_abs_x[:, ni] = ego_x_init + rel_x
            npc_abs_speed[:, ni] = ego_speed_init + rel_speed
            npc_lane[:, ni] = ego_lane_init + rel_lane
            npc_exists[:, ni] = exists

        # Track ego state per rollout (including lane!)
        ego_speeds = np.full(num_rollouts, ego_speed_init)
        ego_xs = np.full(num_rollouts, ego_x_init)
        ego_lanes = np.full(num_rollouts, ego_lane_init, dtype=int)

        total_rewards = np.zeros(num_rollouts)
        discount = 1.0

        # Debug: track per-step rewards for pure-action rollouts (0..8)
        if self.debug:
            debug_per_step = np.zeros((num_actions, horizon))
            debug_ego_speeds = np.zeros((num_actions, horizon))
            debug_ego_lanes = np.zeros((num_actions, horizon), dtype=int)
            debug_ego_xs = np.zeros((num_actions, horizon))

        self.wm.eval()
        first_effective_actions = None
        with torch.no_grad():
            for t in range(horizon):
                prev_speeds = ego_speeds.copy()
                prev_xs = ego_xs.copy()
                prev_lanes = ego_lanes.copy()

                # Step ego with exact kinematics (speed, position, lane)
                # Also compute effective actions (after lane clipping) for WM
                effective_actions = all_actions[:, t].copy()
                for i in range(num_rollouts):
                    action = Action.from_index(all_actions[i, t])
                    accel = self._get_accel(action)
                    ego_speeds[i] = np.clip(
                        ego_speeds[i] + accel * dt,
                        self.vc.min_speed, self.vc.max_speed)
                    ego_xs[i] += ego_speeds[i] * dt
                    lane_delta = self._get_lane_change(action)
                    new_lane = np.clip(
                        ego_lanes[i] + lane_delta, 0, num_lanes - 1)
                    # If lane change was clipped, remap action to keep-lane
                    # so WM sees the action that actually happened
                    if new_lane == ego_lanes[i] and lane_delta != 0:
                        # Map to same longitudinal + KEEP lateral
                        lon_idx = all_actions[i, t] // 3  # 0,1,2
                        effective_actions[i] = lon_idx * 3 + 1  # lateral KEEP=1
                    ego_lanes[i] = new_lane

                # Remember effective actions for t=0 (what we'll actually output)
                if t == 0:
                    first_effective_actions = effective_actions.copy()

                # Build observation for WM (relative coords from current ego)
                obs_for_wm = np.zeros((num_rollouts, obs_dim), dtype=np.float32)
                obs_for_wm[:, 0] = prev_speeds  # ego speed before action
                obs_for_wm[:, 1] = prev_lanes   # ego lane before action
                obs_for_wm[:, 2] = prev_xs
                for ni in range(k):
                    base = 3 + ni * 4
                    obs_for_wm[:, base] = npc_abs_x[:, ni] - prev_xs  # rel_x
                    obs_for_wm[:, base + 1] = npc_lane[:, ni] - prev_lanes  # rel_lane
                    obs_for_wm[:, base + 2] = npc_abs_speed[:, ni] - prev_speeds  # rel_speed
                    obs_for_wm[:, base + 3] = npc_exists[:, ni]

                # Normalize and run WM
                if self.normalizer is not None:
                    obs_norm = self.normalizer.transform(obs_for_wm)
                else:
                    obs_norm = obs_for_wm
                obs_t = torch.FloatTensor(obs_norm).to(self.device)
                actions_t = torch.LongTensor(effective_actions).to(self.device)

                # Get WM prediction (full next obs in normalized space)
                pred_norm = self.wm(obs_t, actions_t)
                if self.normalizer is not None:
                    pred_real = self.normalizer.inverse_transform(
                        pred_norm.cpu().numpy())
                else:
                    pred_real = pred_norm.cpu().numpy()

                # Extract predicted NPC deltas from WM output
                # WM predicts: next_rel_x = rel_x + delta_rel_x
                # delta_rel_x = next_rel_x - rel_x
                #             = (npc_x' - ego_x') - (npc_x - ego_x)
                #             = (npc_x' - npc_x) - (ego_x' - ego_x)
                #             = npc_delta_x - ego_delta_x
                # So: npc_delta_x = delta_rel_x + ego_delta_x
                ego_delta_x = ego_xs - prev_xs
                ego_delta_speed = ego_speeds - prev_speeds

                for ni in range(k):
                    base = 3 + ni * 4
                    pred_rel_x = pred_real[:, base]
                    pred_rel_speed = pred_real[:, base + 2]
                    cur_rel_x = obs_for_wm[:, base]
                    cur_rel_speed = obs_for_wm[:, base + 2]

                    delta_rel_x = pred_rel_x - cur_rel_x
                    delta_rel_speed = pred_rel_speed - cur_rel_speed

                    # Convert to absolute NPC deltas
                    npc_delta_x = delta_rel_x + ego_delta_x
                    npc_delta_speed = delta_rel_speed + ego_delta_speed

                    npc_abs_x[:, ni] += npc_delta_x
                    npc_abs_speed[:, ni] += npc_delta_speed

                # Compute reward from exact ego + predicted NPC positions
                rewards = self._estimate_reward(
                    prev_speeds, ego_speeds, ego_xs, ego_lanes, prev_lanes,
                    npc_abs_x, npc_abs_speed, npc_lane, npc_exists,
                    effective_actions)
                total_rewards += discount * rewards
                discount *= self.mc.planner_discount

                if self.debug:
                    for i in range(num_actions):
                        debug_per_step[i, t] = rewards[i]
                        debug_ego_speeds[i, t] = ego_speeds[i]
                        debug_ego_lanes[i, t] = ego_lanes[i]
                        debug_ego_xs[i, t] = ego_xs[i]

                    # Track NPC positions for KP/- (action index 3) to diagnose hallucinations
                    if not hasattr(self, '_debug_npc_data'):
                        self._debug_npc_data = {}
                    kp_idx = 3  # KP/-
                    self._debug_npc_data[t] = {
                        'ego_x': ego_xs[kp_idx],
                        'ego_lane': ego_lanes[kp_idx],
                        'npc_xs': npc_abs_x[kp_idx].copy(),
                        'npc_lanes': npc_lane[kp_idx].copy(),
                        'npc_exists': npc_exists[kp_idx].copy(),
                        'npc_speeds': npc_abs_speed[kp_idx].copy(),
                    }

        # Tie-breaking: prefer KEEP lane, then random between LEFT/RIGHT
        max_reward = total_rewards.max()
        best_candidates = np.where(np.abs(total_rewards - max_reward) < 1e-6)[0]
        # Check if any candidate is a KEEP-lane action
        keep_candidates = [c for c in best_candidates
                           if Action.from_index(int(first_effective_actions[c])).lateral.value == 0]
        if keep_candidates:
            best_idx = self.rng.choice(keep_candidates)
        else:
            best_idx = self.rng.choice(best_candidates)
        self._step_count += 1

        if self.debug:
            action_names = ['ACC/-', 'ACC/L', 'ACC/R',
                            ' KP/-', ' KP/L', ' KP/R',
                            'BRK/-', 'BRK/L', 'BRK/R']
            print(f"\n=== Step {self._step_count} | ego_spd={ego_speed_init:.1f} lane={ego_lane_init} ===")

            # Summary table
            print(f"  {'Action':>6} | {'Total':>8} | per-step rewards")
            for i in range(num_actions):
                steps_str = " ".join(f"{debug_per_step[i,t]:+.3f}" for t in range(horizon))
                marker = " <== BEST" if i == best_idx else ""
                print(f"  {action_names[i]:>6} | {total_rewards[i]:>8.3f} | [{steps_str}]{marker}")

            # Detailed comparison: KP/- vs lane change actions
            keep_idx = 3  # KP/-
            for lc_idx in [2, 5, 8, 1, 4, 7]:  # ACC/R, KP/R, BRK/R, ACC/L, KP/L, BRK/L
                lc_name = action_names[lc_idx]
                diff = total_rewards[lc_idx] - total_rewards[keep_idx]
                if abs(diff) < 5.0:  # only show relevant comparisons
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

            # Best random rollout
            if best_idx >= num_actions:
                first_act = int(all_actions[best_idx, 0])
                print(f"\n  Best is random rollout #{best_idx}, first_act={action_names[first_act]}, "
                      f"reward={total_rewards[best_idx]:.3f}")
            print(f"  Chosen: {action_names[int(all_actions[best_idx, 0])]}")

            # Dump WM-predicted NPC positions for KP/- if it has a bad step
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
            if hasattr(self, '_debug_npc_data'):
                del self._debug_npc_data

        chosen = Action.from_index(int(first_effective_actions[best_idx]))

        # Lane-change cooldown: prevent oscillation by blocking lane changes
        # within N steps of the last one
        if chosen.lateral.value != 0:
            steps_since = self._step_count - self._last_lane_change_step
            if steps_since < self._lane_change_cooldown:
                # Force keep-lane, preserve longitudinal
                chosen = Action(chosen.longitudinal, LateralAction.KEEP)
            else:
                self._last_lane_change_step = self._step_count

        # Safety override using REAL observation (not WM predictions).
        # The WM may hallucinate safe gaps that don't exist.
        chosen = self._apply_safety(obs, chosen)

        return chosen

    def _apply_safety(self, obs: np.ndarray, action: Action) -> Action:
        """Override dangerous actions based on current real observation.

        Two modes:
        - Normal: block unsafe lane changes (target gap < 15m)
        - Emergency: if same-lane gap is critically small, allow lane
          changes even with tighter target gaps (but still > 5m)
        """
        ego_speed = obs[0]
        ego_lane = int(round(obs[1]))
        k = (len(obs) - 3) // 4
        lateral = action.lateral.value
        target_lane = ego_lane + lateral

        # Scan all NPCs
        min_gap_ahead = float('inf')
        min_target_ahead = float('inf')
        min_target_behind = float('inf')

        for i in range(k):
            base = 3 + i * 4
            rel_x = obs[base]
            rel_lane = obs[base + 1]
            exists = obs[base + 3]
            if exists < 0.5:
                continue
            npc_lane = ego_lane + rel_lane
            # Same-lane ahead
            if abs(rel_lane) < 0.5 and rel_x > 0:
                min_gap_ahead = min(min_gap_ahead, rel_x)
            # Target lane for lane changes
            if lateral != 0 and abs(npc_lane - target_lane) < 0.5:
                if rel_x > 0:
                    min_target_ahead = min(min_target_ahead, rel_x)
                else:
                    min_target_behind = min(min_target_behind, abs(rel_x))

        # Determine if we're in an emergency (about to collide in current lane)
        emergency = min_gap_ahead < 10.0

        # Block unsafe lane changes
        if lateral != 0:
            if emergency:
                # Emergency: allow tighter merges, but still require 5m clearance
                if min_target_ahead < 5.0 or min_target_behind < 5.0:
                    action = Action(action.longitudinal, LateralAction.KEEP)
            else:
                # Normal: require 15m clearance
                if min_target_ahead < 15.0 or min_target_behind < 15.0:
                    action = Action(action.longitudinal, LateralAction.KEEP)

        # Force brake if same-lane gap ahead < safe following distance
        safe_gap = ego_speed * 0.5
        if min_gap_ahead < safe_gap:
            if action.longitudinal != LongitudinalAction.DECELERATE:
                action = Action(LongitudinalAction.DECELERATE, action.lateral)

        # Force brake if very close
        if min_gap_ahead < 10.0:
            action = Action(LongitudinalAction.DECELERATE, action.lateral)

        return action

    def _estimate_reward(self, prev_speeds, ego_speeds, ego_xs,
                         ego_lanes, prev_lanes,
                         npc_xs, npc_speeds, npc_lanes, npc_exists,
                         actions) -> np.ndarray:
        """Estimate reward with clean separation of safety vs optimization.

        Logic:
        1. SAFETY: Is the action catastrophically unsafe?
           - Collision (same-lane gap < 5m) → -100
           - Lane change into occupied space (gap < 15m) → -100
           - Too close to car ahead given current speed → reward = 0
             (not catastrophic, but clearly worse than any positive speed reward)
        2. OPTIMIZATION: Among safe actions, which is best?
           - speed / speed_limit, capped at 1.0
        """
        batch = len(ego_speeds)
        rewards = np.zeros(batch)

        # === OPTIMIZATION: Speed reward ===
        speed_limit = max(self.rc.speed_limit, 1e-6)
        for i in range(batch):
            if ego_speeds[i] <= speed_limit:
                rewards[i] = ego_speeds[i] / speed_limit
            else:
                excess = ego_speeds[i] - speed_limit
                rewards[i] = 1.0 - excess / 5.0

        # Small lane-change cost: just enough to prefer KEEP when
        # speed reward is similar, but not enough to prevent useful passes.
        # At 30 m/s, speed reward = 1.0, so 0.01 is ~1% — negligible
        # for decision-making but breaks near-ties in favor of staying.
        changed_lane_action = (ego_lanes != prev_lanes)
        rewards[changed_lane_action] -= 0.01

        # === SAFETY CHECKS ===
        k = npc_xs.shape[1]
        for n in range(k):
            raw_gap = npc_xs[:, n] - ego_xs
            same_lane = (np.abs(npc_lanes[:, n] - ego_lanes) < 0.5) & \
                        (npc_exists[:, n] > 0.5)

            # Check 1: Collision (gap < 5m)
            collision = same_lane & (np.abs(raw_gap) < 5.0)
            rewards[collision] = -100.0

            # Check 2: Too close ahead — need safe following distance
            # At 30 m/s with 5 m/s² braking, stopping takes 3s = 45m relative.
            # But NPC is also moving, so relative closing matters.
            # Simple rule: maintain at least 1 second of following distance
            # based on speed difference (closing rate).
            # If gap < ego_speed * 0.5s (half-second rule), reward = 0.
            # This is ~15m at 30 m/s, ~10m at 20 m/s. Physics-based, not magic.
            ahead = same_lane & (raw_gap > 0)
            min_gap = ego_speeds * 0.5  # half-second following distance
            too_close = ahead & (raw_gap < min_gap)
            rewards[too_close] = np.minimum(rewards[too_close], 0.0)

        # Check 3: Lane change into occupied space
        # Ahead gap < 15m or behind gap < 10m → unsafe
        changed_lane = (ego_lanes != prev_lanes)
        if np.any(changed_lane):
            for n in range(k):
                if not np.any(npc_exists[:, n] > 0.5):
                    continue
                in_new_lane = (np.abs(npc_lanes[:, n] - ego_lanes) < 0.5) & \
                              (npc_exists[:, n] > 0.5) & changed_lane
                if not np.any(in_new_lane):
                    continue
                gap = npc_xs[:, n] - ego_xs  # positive = NPC ahead
                unsafe_ahead = in_new_lane & (gap > 0) & (gap < 15.0)
                unsafe_behind = in_new_lane & (gap <= 0) & (np.abs(gap) < 10.0)
                rewards[unsafe_ahead | unsafe_behind] = -100.0

        return rewards
