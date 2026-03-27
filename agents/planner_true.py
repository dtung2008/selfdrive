"""MPC-style planner using the TRUE simulator for rollouts.

This module implements a Model Predictive Control (MPC) planner that has
access to the actual simulator for lookahead. Because it uses the real
simulator (clone/restore), its rollouts are perfectly accurate -- there
is no model error. This serves as an upper-bound baseline for planners
that must rely on learned world models.

The planner evaluates three categories of candidate action sequences:
  1. Hand-crafted driving maneuvers (e.g., pass-left, escape-right)
  2. Single-action repeats (each of the 9 actions held for the full horizon)
  3. Random action sequences (for exploration diversity)

Only the first action of the best-scoring sequence is executed (receding
horizon control), and the process repeats at the next timestep.
"""
import numpy as np
from agents.base import Agent
from utils.types import Action, LongitudinalAction, LateralAction
from utils.config import ModelConfig


# ---------------------------------------------------------------------------
# Pre-built driving maneuvers
# ---------------------------------------------------------------------------
# Each maneuver is a callable: horizon (int) -> np.ndarray of action indices.
# These encode common driving behaviors as fixed action sequences so the
# planner does not have to discover them purely by random sampling.
# The lane-change maneuvers apply the lateral action only on the first step,
# then revert to straight-lane driving for the remainder of the horizon.
# ---------------------------------------------------------------------------

def _maneuver_keep_speed(horizon):
    """Maintain current speed and lane for the entire horizon.

    Args:
        horizon: Number of planning steps.

    Returns:
        A numpy array of action indices (all KEEP/KEEP).
    """
    return np.full(horizon, Action(LongitudinalAction.KEEP, LateralAction.KEEP).to_index())

def _maneuver_accelerate(horizon):
    """Accelerate in the current lane for the entire horizon.

    Args:
        horizon: Number of planning steps.

    Returns:
        A numpy array of action indices (all ACCELERATE/KEEP).
    """
    return np.full(horizon, Action(LongitudinalAction.ACCELERATE, LateralAction.KEEP).to_index())

def _maneuver_brake(horizon):
    """Brake in the current lane for the entire horizon.

    Args:
        horizon: Number of planning steps.

    Returns:
        A numpy array of action indices (all DECELERATE/KEEP).
    """
    return np.full(horizon, Action(LongitudinalAction.DECELERATE, LateralAction.KEEP).to_index())

def _maneuver_pass_left(horizon):
    """Overtake by changing left then accelerating straight.

    This represents a passing maneuver: move one lane left on step 0
    while accelerating, then continue accelerating straight ahead.

    Args:
        horizon: Number of planning steps.

    Returns:
        A numpy array of action indices.
    """
    seq = np.full(horizon, Action(LongitudinalAction.ACCELERATE, LateralAction.KEEP).to_index())
    seq[0] = Action(LongitudinalAction.ACCELERATE, LateralAction.LEFT).to_index()
    return seq

def _maneuver_pass_right(horizon):
    """Overtake by changing right then accelerating straight.

    Mirror of _maneuver_pass_left for right-side passing.

    Args:
        horizon: Number of planning steps.

    Returns:
        A numpy array of action indices.
    """
    seq = np.full(horizon, Action(LongitudinalAction.ACCELERATE, LateralAction.KEEP).to_index())
    seq[0] = Action(LongitudinalAction.ACCELERATE, LateralAction.RIGHT).to_index()
    return seq

def _maneuver_escape_left(horizon):
    """Dodge left while maintaining speed.

    Useful for evading a slow or stopped vehicle: change lane left on
    step 0, then cruise straight at the current speed.

    Args:
        horizon: Number of planning steps.

    Returns:
        A numpy array of action indices.
    """
    seq = np.full(horizon, Action(LongitudinalAction.KEEP, LateralAction.KEEP).to_index())
    seq[0] = Action(LongitudinalAction.KEEP, LateralAction.LEFT).to_index()
    return seq

def _maneuver_escape_right(horizon):
    """Dodge right while maintaining speed.

    Mirror of _maneuver_escape_left for right-side evasion.

    Args:
        horizon: Number of planning steps.

    Returns:
        A numpy array of action indices.
    """
    seq = np.full(horizon, Action(LongitudinalAction.KEEP, LateralAction.KEEP).to_index())
    seq[0] = Action(LongitudinalAction.KEEP, LateralAction.RIGHT).to_index()
    return seq

def _maneuver_brake_left(horizon):
    """Brake while changing lane left.

    Combines deceleration with a left lane change on step 0, useful
    for emergency avoidance where both braking and dodging are needed.

    Args:
        horizon: Number of planning steps.

    Returns:
        A numpy array of action indices.
    """
    seq = np.full(horizon, Action(LongitudinalAction.DECELERATE, LateralAction.KEEP).to_index())
    seq[0] = Action(LongitudinalAction.DECELERATE, LateralAction.LEFT).to_index()
    return seq

def _maneuver_brake_right(horizon):
    """Brake while changing lane right.

    Mirror of _maneuver_brake_left for right-side evasion.

    Args:
        horizon: Number of planning steps.

    Returns:
        A numpy array of action indices.
    """
    seq = np.full(horizon, Action(LongitudinalAction.DECELERATE, LateralAction.KEEP).to_index())
    seq[0] = Action(LongitudinalAction.DECELERATE, LateralAction.RIGHT).to_index()
    return seq


# Registry of all hand-crafted maneuvers. These are evaluated first during
# planning, before single-action repeats and random sequences.
MANEUVERS = [
    _maneuver_keep_speed,
    _maneuver_accelerate,
    _maneuver_brake,
    _maneuver_pass_left,
    _maneuver_pass_right,
    _maneuver_escape_left,
    _maneuver_escape_right,
    _maneuver_brake_left,
    _maneuver_brake_right,
]


class PlannerTrueModel(Agent):
    """MPC planner that uses the real simulator's clone/restore for lookahead.

    This planner has access to the ground-truth simulator dynamics. At each
    decision step it saves the simulator state, evaluates many candidate
    action sequences by actually stepping the simulator forward, then
    restores the original state and executes only the first action of the
    best sequence (receding horizon / MPC).

    The candidate action sequences are structured into three tiers:
      1. Hand-crafted driving maneuvers (MANEUVERS list above)
      2. Single-action repeats (each of the 9 discrete actions repeated
         for the full planning horizon)
      3. Random action sequences (to fill the remaining rollout budget)

    Tie-breaking favors lane-keeping actions over lane changes to promote
    stable, predictable driving behavior.

    Attributes:
        sim: The simulator instance, which must support clone_state(),
            restore_state(), and step() methods.
        mc: ModelConfig containing planner_horizon, planner_num_rollouts,
            and planner_discount parameters.
    """

    def __init__(self, simulator, model_config: ModelConfig = None,
                 seed: int = 42):
        """Initialize the true-model planner.

        Args:
            simulator: A simulator instance with clone_state(),
                restore_state(snapshot), and step(action) methods.
            model_config: Configuration object specifying planning horizon,
                number of rollouts, and discount factor. Uses defaults
                if None.
            seed: Random seed for reproducible action sampling.
        """
        self.sim = simulator
        self.mc = model_config or ModelConfig()
        self.sample_rng = np.random.RandomState(seed)
        self.select_rng = np.random.RandomState(seed + 1)

    def act(self, obs: np.ndarray) -> Action:
        """Select the best action via MPC with structured and random rollouts.

        Saves the simulator state, evaluates all candidate action sequences
        by stepping the simulator forward over the planning horizon, then
        restores the original state and returns the first action from the
        highest-scoring sequence.

        Args:
            obs: The current observation vector (not directly used; the
                simulator's internal state is used for rollouts instead).

        Returns:
            The first Action from the best-scoring candidate sequence.
        """
        # Save the current simulator state so we can restore after rollouts
        snapshot = self.sim.clone_state()
        num_actions = Action.num_actions()
        horizon = self.mc.planner_horizon
        num_rollouts = self.mc.planner_num_rollouts

        # Build candidate action sequences from three sources
        candidates = []

        # Tier 1: Hand-crafted driving maneuvers that encode common
        # multi-step behaviors (passing, braking, dodging, etc.)
        for maneuver in MANEUVERS:
            candidates.append(maneuver(horizon))

        # Tier 2: Single-action repeats -- each of the 9 discrete actions
        # held constant for the full horizon. This ensures every primitive
        # action is evaluated even if no maneuver includes it.
        for a in range(num_actions):
            candidates.append(np.full(horizon, a, dtype=int))

        # Tier 3: Random sequences to fill the remaining rollout budget.
        # These provide exploration diversity beyond the structured candidates.
        while len(candidates) < num_rollouts:
            candidates.append(self.sample_rng.randint(0, num_actions, size=horizon))

        # Evaluate each candidate by rolling out in the real simulator
        all_rewards = []
        all_first_actions = []
        for action_indices in candidates:
            # Restore to the saved state before each rollout so every
            # candidate starts from the same initial conditions
            self.sim.restore_state(snapshot)
            total_reward = 0.0
            discount = 1.0
            for t, aidx in enumerate(action_indices):
                action = Action.from_index(int(aidx))
                _, reward, done, _ = self.sim.step(action)
                # Accumulate discounted reward over the horizon
                total_reward += discount * reward
                discount *= self.mc.planner_discount
                # Stop early if the episode terminates (e.g., collision)
                if done:
                    break
            all_rewards.append(total_reward)
            all_first_actions.append(Action.from_index(int(action_indices[0])))

        # Select the best candidate with tie-breaking logic:
        # Among candidates with the maximum reward (within floating-point
        # tolerance), prefer those whose first action keeps the current lane.
        # This reduces unnecessary lane oscillation.
        max_reward = max(all_rewards)
        best_candidates = [i for i, r in enumerate(all_rewards)
                           if abs(r - max_reward) < 1e-6]
        keep_candidates = [i for i in best_candidates
                           if all_first_actions[i].lateral.value == 0]
        if keep_candidates:
            best_idx = self.select_rng.choice(keep_candidates)
        else:
            best_idx = self.select_rng.choice(best_candidates)

        # Restore the simulator to its original state before returning,
        # since only the first action will actually be applied externally
        self.sim.restore_state(snapshot)
        return all_first_actions[best_idx]
