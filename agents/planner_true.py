"""MPC-style planner using the TRUE simulator for rollouts."""
import numpy as np
from agents.base import Agent
from utils.types import Action, LongitudinalAction, LateralAction
from utils.config import ModelConfig


# Pre-built driving maneuvers: (description, action sequence builder)
# Each maneuver is a function: horizon -> list of action indices

def _maneuver_keep_speed(horizon):
    """Maintain current speed and lane."""
    return np.full(horizon, Action(LongitudinalAction.KEEP, LateralAction.KEEP).to_index())

def _maneuver_accelerate(horizon):
    """Accelerate in current lane."""
    return np.full(horizon, Action(LongitudinalAction.ACCELERATE, LateralAction.KEEP).to_index())

def _maneuver_brake(horizon):
    """Brake in current lane."""
    return np.full(horizon, Action(LongitudinalAction.DECELERATE, LateralAction.KEEP).to_index())

def _maneuver_pass_left(horizon):
    """Lane change left then accelerate."""
    seq = np.full(horizon, Action(LongitudinalAction.ACCELERATE, LateralAction.KEEP).to_index())
    seq[0] = Action(LongitudinalAction.ACCELERATE, LateralAction.LEFT).to_index()
    return seq

def _maneuver_pass_right(horizon):
    """Lane change right then accelerate."""
    seq = np.full(horizon, Action(LongitudinalAction.ACCELERATE, LateralAction.KEEP).to_index())
    seq[0] = Action(LongitudinalAction.ACCELERATE, LateralAction.RIGHT).to_index()
    return seq

def _maneuver_escape_left(horizon):
    """Lane change left and keep speed (dodge)."""
    seq = np.full(horizon, Action(LongitudinalAction.KEEP, LateralAction.KEEP).to_index())
    seq[0] = Action(LongitudinalAction.KEEP, LateralAction.LEFT).to_index()
    return seq

def _maneuver_escape_right(horizon):
    """Lane change right and keep speed (dodge)."""
    seq = np.full(horizon, Action(LongitudinalAction.KEEP, LateralAction.KEEP).to_index())
    seq[0] = Action(LongitudinalAction.KEEP, LateralAction.RIGHT).to_index()
    return seq

def _maneuver_brake_left(horizon):
    """Brake and lane change left."""
    seq = np.full(horizon, Action(LongitudinalAction.DECELERATE, LateralAction.KEEP).to_index())
    seq[0] = Action(LongitudinalAction.DECELERATE, LateralAction.LEFT).to_index()
    return seq

def _maneuver_brake_right(horizon):
    """Brake and lane change right."""
    seq = np.full(horizon, Action(LongitudinalAction.DECELERATE, LateralAction.KEEP).to_index())
    seq[0] = Action(LongitudinalAction.DECELERATE, LateralAction.RIGHT).to_index()
    return seq


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
    """Planner that uses the real simulator's clone/restore for lookahead.

    Uses structured rollouts:
      1. Driving maneuvers (pass left, pass right, brake, accelerate, etc.)
      2. Single-action repeats (all 9 actions)
      3. Random sequences for diversity
    """

    def __init__(self, simulator, model_config: ModelConfig = None,
                 seed: int = 42):
        self.sim = simulator
        self.mc = model_config or ModelConfig()
        self.rng = np.random.RandomState(seed)

    def act(self, obs: np.ndarray) -> Action:
        """Run MPC with structured + random rollouts."""
        snapshot = self.sim.clone_state()
        best_reward = -float("inf")
        best_first_action = Action()
        num_actions = Action.num_actions()
        horizon = self.mc.planner_horizon
        num_rollouts = self.mc.planner_num_rollouts

        # Build candidate action sequences
        candidates = []

        # 1. Driving maneuvers
        for maneuver in MANEUVERS:
            candidates.append(maneuver(horizon))

        # 2. Single-action repeats
        for a in range(num_actions):
            candidates.append(np.full(horizon, a, dtype=int))

        # 3. Fill remaining with random
        while len(candidates) < num_rollouts:
            candidates.append(self.rng.randint(0, num_actions, size=horizon))

        # Evaluate each candidate
        all_rewards = []
        all_first_actions = []
        for action_indices in candidates:
            self.sim.restore_state(snapshot)
            total_reward = 0.0
            discount = 1.0
            for t, aidx in enumerate(action_indices):
                action = Action.from_index(int(aidx))
                _, reward, done, _ = self.sim.step(action)
                total_reward += discount * reward
                discount *= self.mc.planner_discount
                if done:
                    break
            all_rewards.append(total_reward)
            all_first_actions.append(Action.from_index(int(action_indices[0])))

        # Tie-breaking: prefer KEEP, then random among LEFT/RIGHT
        max_reward = max(all_rewards)
        best_candidates = [i for i, r in enumerate(all_rewards)
                           if abs(r - max_reward) < 1e-6]
        keep_candidates = [i for i in best_candidates
                           if all_first_actions[i].lateral.value == 0]
        if keep_candidates:
            best_idx = self.rng.choice(keep_candidates)
        else:
            best_idx = self.rng.choice(best_candidates)

        self.sim.restore_state(snapshot)
        return all_first_actions[best_idx]
