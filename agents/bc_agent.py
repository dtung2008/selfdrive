"""Behavior cloning agent: neural network imitating the expert."""
import torch
import numpy as np
from agents.base import Agent
from models.policy_net import PolicyNetwork
from utils.types import Action


class BCAgent(Agent):
    """Behaviour cloning agent — wraps a trained PolicyNetwork.

    Includes an optional safety override: if the chosen action would
    accelerate or keep speed when there's a same-lane NPC very close
    ahead, override to brake.
    """

    def __init__(self, policy: PolicyNetwork, deterministic: bool = True,
                 normalizer=None, safety_override: bool = True):
        self.policy = policy
        self.deterministic = deterministic
        self.normalizer = normalizer
        self.safety_override = safety_override
        self.device = next(policy.parameters()).device

    def act(self, obs: np.ndarray) -> Action:
        if self.normalizer is not None:
            obs_norm = self.normalizer.normalize(obs)
        else:
            obs_norm = obs
        obs_t = torch.FloatTensor(obs_norm).unsqueeze(0).to(self.device)
        if self.deterministic:
            idx = self.policy.greedy_action(obs_t)
        else:
            idx = self.policy.sample_action(obs_t)
        action = Action.from_index(idx)

        if self.safety_override:
            action = self._apply_safety(obs, action)

        return action

    def _apply_safety(self, obs: np.ndarray, action: Action) -> Action:
        """Override dangerous actions based on current observation."""
        from utils.types import LongitudinalAction, LateralAction

        ego_speed = obs[0]
        ego_lane = int(round(obs[1]))
        k = (len(obs) - 3) // 4

        # Find closest same-lane NPC ahead, and check target lane gaps
        min_gap_ahead = float('inf')
        lateral = action.lateral.value  # -1=left, 0=keep, +1=right
        target_lane = ego_lane + lateral
        min_target_ahead = float('inf')   # closest NPC ahead in target lane
        min_target_behind = float('inf')  # closest NPC behind in target lane

        for i in range(k):
            base = 3 + i * 4
            rel_x = obs[base]
            rel_lane = obs[base + 1]
            exists = obs[base + 3]
            if exists < 0.5:
                continue
            npc_lane = ego_lane + rel_lane
            # Same-lane ahead check
            if abs(rel_lane) < 0.5 and rel_x > 0:
                min_gap_ahead = min(min_gap_ahead, rel_x)
            # Target lane check (for lane changes)
            if lateral != 0 and abs(npc_lane - target_lane) < 0.5:
                if rel_x > 0:
                    min_target_ahead = min(min_target_ahead, rel_x)
                else:
                    min_target_behind = min(min_target_behind, abs(rel_x))

        # Determine if we're in an emergency
        emergency = min_gap_ahead < 10.0

        # Block unsafe lane changes
        if lateral != 0:
            if emergency:
                # Emergency: allow tighter merges but still require 5m
                if min_target_ahead < 5.0 or min_target_behind < 5.0:
                    action = Action(action.longitudinal, LateralAction.KEEP)
            else:
                # Normal: require 15m clearance
                if min_target_ahead < 15.0 or min_target_behind < 15.0:
                    action = Action(action.longitudinal, LateralAction.KEEP)

        # If gap ahead < safe following distance, force brake
        safe_gap = ego_speed * 0.5  # half-second rule
        if min_gap_ahead < safe_gap:
            if action.longitudinal != LongitudinalAction.DECELERATE:
                return Action(LongitudinalAction.DECELERATE, action.lateral)

        # If gap ahead < 10m, force brake
        if min_gap_ahead < 10.0:
            return Action(LongitudinalAction.DECELERATE, action.lateral)

        return action
