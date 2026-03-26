"""Behavior cloning agent: neural network imitating the expert.

This module implements a behavior cloning (BC) agent that wraps a trained
PolicyNetwork to produce driving actions from observations. Behavior cloning
is a supervised-learning approach to imitation learning where the agent
learns to map observations directly to actions by mimicking expert
demonstrations.

The agent also includes a rule-based safety override layer that can veto
dangerous actions (e.g., accelerating into a vehicle directly ahead) and
replace them with safer alternatives (e.g., braking). This safety layer
operates on the raw observation, independent of the neural network.
"""
import torch
import numpy as np
from agents.base import Agent
from models.policy_net import PolicyNetwork
from utils.types import Action


class BCAgent(Agent):
    """Behaviour cloning agent -- wraps a trained PolicyNetwork.

    This agent queries a neural network policy to select actions, then
    optionally applies a rule-based safety override. The safety override
    inspects the raw observation to detect imminent collision risks and
    forces braking or blocks unsafe lane changes when necessary.

    The observation vector layout is assumed to be:
        [ego_speed, ego_lane, ego_x,
         npc0_rel_x, npc0_rel_lane, npc0_rel_speed, npc0_exists,
         npc1_rel_x, npc1_rel_lane, npc1_rel_speed, npc1_exists, ...]

    Attributes:
        policy: The trained PolicyNetwork that maps observations to actions.
        deterministic: If True, use greedy (argmax) action selection;
            otherwise sample from the policy's output distribution.
        normalizer: Optional observation normalizer (e.g., z-score scaler).
            Applied before feeding observations to the neural network.
        safety_override: If True, apply rule-based safety checks after
            the network selects an action.
        device: The torch device (CPU/GPU) on which the policy resides.
    """

    def __init__(self, policy: PolicyNetwork, deterministic: bool = True,
                 normalizer=None, safety_override: bool = True):
        """Initialize the behavior cloning agent.

        Args:
            policy: A trained PolicyNetwork instance.
            deterministic: Whether to use greedy action selection (True) or
                stochastic sampling (False).
            normalizer: Optional normalizer object with a .normalize() method
                that transforms raw observations before policy inference.
            safety_override: Whether to apply rule-based safety overrides
                after the policy selects an action.
        """
        self.policy = policy
        self.deterministic = deterministic
        self.normalizer = normalizer
        self.safety_override = safety_override
        # Infer the device from the policy's parameters so tensors are
        # created on the correct device (CPU or GPU)
        self.device = next(policy.parameters()).device

    def act(self, obs: np.ndarray) -> Action:
        """Select a driving action given the current observation.

        The observation is optionally normalized, converted to a torch tensor,
        and passed through the policy network to produce an action index.
        If safety_override is enabled, the chosen action may be replaced
        by a safer alternative.

        Args:
            obs: A 1-D numpy array containing the raw observation vector.

        Returns:
            An Action object representing the chosen longitudinal and
            lateral driving action.
        """
        # Normalize observation if a normalizer was provided
        if self.normalizer is not None:
            obs_norm = self.normalizer.normalize(obs)
        else:
            obs_norm = obs

        # Convert to a batched torch tensor (batch_size=1) on the correct device
        obs_t = torch.FloatTensor(obs_norm).unsqueeze(0).to(self.device)

        # Select action index from the policy network
        if self.deterministic:
            idx = self.policy.greedy_action(obs_t)
        else:
            idx = self.policy.sample_action(obs_t)

        # Convert integer index back to a structured Action object
        action = Action.from_index(idx)

        # Optionally apply safety override using the RAW (un-normalized)
        # observation so that physical distances are meaningful
        if self.safety_override:
            action = self._apply_safety(obs, action)

        return action

    def _apply_safety(self, obs: np.ndarray, action: Action) -> Action:
        """Override dangerous actions based on the current observation.

        This method scans all NPC vehicles visible in the observation to
        compute gap distances. It enforces two categories of safety rules:

        1. Lane-change blocking: Prevents lane changes when there is
           insufficient clearance in the target lane. In emergency mode
           (same-lane gap < 10m), clearance thresholds are relaxed to
           allow evasive maneuvers.

        2. Three-state following distance: Uses physics-based stopping
           distance (closing_speed² / 2a + buffer) when closing on a
           slower vehicle, caps at KEEP when gap is small but stable,
           and always brakes below 10m absolute minimum.

        Args:
            obs: The raw (un-normalized) observation vector.
            action: The action originally chosen by the policy network.

        Returns:
            The (possibly overridden) Action. If the original action was
            safe, it is returned unchanged.
        """
        from utils.types import LongitudinalAction, LateralAction

        # Extract ego vehicle state from the observation
        ego_speed = obs[0]
        ego_lane = int(round(obs[1]))

        # Calculate the number of NPC vehicles in the observation.
        # Each NPC occupies 4 fields: [rel_x, rel_lane, rel_speed, exists].
        # The first 3 fields are ego state, so k = (obs_dim - 3) / 4.
        k = (len(obs) - 3) // 4

        # Determine the target lane if a lane change is requested
        min_gap_ahead = float('inf')
        lateral = action.lateral.value  # -1=left, 0=keep, +1=right
        target_lane = ego_lane + lateral
        min_target_ahead = float('inf')   # closest NPC ahead in target lane
        min_target_behind = float('inf')  # closest NPC behind in target lane

        # Scan all NPC slots to find the closest relevant vehicles.
        # Also track closing speed to the nearest same-lane NPC ahead for
        # physics-based stopping distance calculation.
        closing_speed_ahead = 0.0  # positive = ego approaching NPC
        for i in range(k):
            base = 3 + i * 4
            rel_x = obs[base]          # longitudinal distance (positive = ahead)
            rel_lane = obs[base + 1]   # lane offset from ego
            rel_speed = obs[base + 2]  # npc_speed - ego_speed (negative = closing)
            exists = obs[base + 3]     # 1.0 if this NPC slot is occupied

            # Skip empty NPC slots (exists acts as a validity flag)
            if exists < 0.5:
                continue

            npc_lane = ego_lane + rel_lane

            # Same-lane ahead check: track the closest vehicle and its
            # closing speed
            if abs(rel_lane) < 0.5 and rel_x > 0:
                if rel_x < min_gap_ahead:
                    min_gap_ahead = rel_x
                    closing_speed_ahead = max(-rel_speed, 0.0)

            # Target lane check (only relevant if a lane change is requested):
            # track the closest vehicle ahead and behind in the lane we want
            # to merge into
            if lateral != 0 and abs(npc_lane - target_lane) < 0.5:
                if rel_x > 0:
                    min_target_ahead = min(min_target_ahead, rel_x)
                else:
                    min_target_behind = min(min_target_behind, abs(rel_x))

        # Emergency mode: vehicle directly ahead is dangerously close (<10m)
        emergency = min_gap_ahead < 10.0

        # Block unsafe lane changes based on target lane clearance
        if lateral != 0:
            if emergency:
                # In emergency, allow tighter merges (evasive action) but
                # still require a minimum 5m gap to avoid side collisions
                if min_target_ahead < 5.0 or min_target_behind < 5.0:
                    action = Action(action.longitudinal, LateralAction.KEEP)
            else:
                # In normal driving, require 15m clearance for lane changes
                # to ensure comfortable and safe merging
                if min_target_ahead < 15.0 or min_target_behind < 15.0:
                    action = Action(action.longitudinal, LateralAction.KEEP)

        # Three-state following distance logic (same-lane only), matching
        # the expert agent's physics-based approach:
        #
        # State 1 - CLOSING: Use physics-based stopping distance to decide
        #   when to brake. Accounts for closing speed via kinematic formula.
        # State 2 - STABLE: Gap is small but not closing. Cap at KEEP to
        #   prevent re-accelerating, but don't force braking.
        # State 3 - CRITICAL: Gap < 10m — always brake.
        max_decel = 5.0  # VehicleConfig.max_deceleration

        if closing_speed_ahead > 0.5:
            # Closing: physics-based safe distance
            stopping_dist = (closing_speed_ahead ** 2) / (2 * max_decel) + 10.0
            safe_gap = max(ego_speed * 0.7, 20.0, stopping_dist)
            if min_gap_ahead < safe_gap:
                if action.longitudinal != LongitudinalAction.DECELERATE:
                    return Action(LongitudinalAction.DECELERATE, action.lateral)
        else:
            # Not closing: prevent acceleration if gap is still small
            if min_gap_ahead < 20.0:
                if action.longitudinal == LongitudinalAction.ACCELERATE:
                    return Action(LongitudinalAction.KEEP, action.lateral)

        # Absolute minimum distance override: always brake if gap < 10m
        if min_gap_ahead < 10.0:
            return Action(LongitudinalAction.DECELERATE, action.lateral)

        return action
