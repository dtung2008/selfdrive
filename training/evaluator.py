"""Evaluate agents and produce comparison metrics.

This module provides a standardised evaluation harness for measuring driving
agent performance.  The :class:`Evaluator` runs an agent for a configurable
number of episodes, recording safety metrics (collision rate), efficiency
metrics (average speed, episode length), behaviour metrics (lane changes), and
overall quality (cumulative reward, completion rate).

The :meth:`Evaluator.compare` convenience method evaluates multiple agents on
the *same* simulator configuration and returns a dictionary of
:class:`EvalMetrics` suitable for tabular comparison.
"""
from typing import Dict, List
from dataclasses import dataclass, field
import numpy as np
from agents.base import Agent
from environment.simulator import Simulator
from utils.types import Action, LateralAction


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics for one agent over multiple episodes.

    All values are scalars computed by averaging (or counting) across episodes.

    Attributes:
        avg_reward:              Mean cumulative reward across episodes.
        std_reward:              Standard deviation of cumulative rewards.
        avg_speed:               Mean speed across all timesteps in all episodes.
        collision_rate:          Fraction of episodes that ended in a collision.
        avg_episode_length:      Mean number of timesteps per episode.
        lane_changes_per_episode: Mean number of lane changes per episode.
        completion_rate:         Fraction of episodes that survived to the
                                 timeout horizon (i.e., did not crash early).
    """
    avg_reward: float = 0.0
    std_reward: float = 0.0
    avg_speed: float = 0.0
    collision_rate: float = 0.0
    avg_episode_length: float = 0.0
    lane_changes_per_episode: float = 0.0
    completion_rate: float = 0.0  # fraction of episodes reaching timeout


class Evaluator:
    """Run an agent for N episodes and compute aggregated metrics.

    Usage::

        evaluator = Evaluator(simulator, num_episodes=50)
        metrics = evaluator.evaluate(my_agent)
        print(metrics)
    """

    def __init__(self, simulator: Simulator, num_episodes: int = 20):
        """Initialise the evaluator.

        Args:
            simulator:    The driving-environment simulator instance.
            num_episodes: Number of evaluation episodes to run per agent.
                          More episodes reduce variance in the reported
                          metrics, but take longer to execute.
        """
        self.sim = simulator
        self.num_episodes = num_episodes

    def evaluate(self, agent: Agent) -> EvalMetrics:
        """Evaluate an agent over multiple episodes and return aggregated metrics.

        Each episode resets both the simulator and the agent, then runs until
        ``done`` is signalled.  Per-step and per-episode statistics are
        accumulated and averaged into an :class:`EvalMetrics` dataclass.

        Args:
            agent: The agent to evaluate.

        Returns:
            An :class:`EvalMetrics` instance containing averaged results.
        """
        total_rewards = []       # Cumulative reward per episode.
        speeds = []              # Speed at every timestep across all episodes.
        collisions = 0           # Count of episodes ending in collision.
        lengths = []             # Timestep count per episode.
        lane_changes_all = []    # Lane-change count per episode.
        completions = 0          # Count of episodes reaching the timeout.

        for _ in range(self.num_episodes):
            obs = self.sim.reset()
            agent.reset()
            done = False
            ep_reward = 0.0
            ep_speeds = []
            ep_lane_changes = 0
            steps = 0

            while not done:
                # Record lane *before* the step to detect lane changes.
                prev_lane = self.sim.ego.lane
                action = agent.act(obs)
                obs, reward, done, info = self.sim.step(action)
                ep_reward += reward
                # Fetch the ego vehicle state to read its current speed
                # and lane, which are not directly part of the observation.
                ego, _ = self.sim.get_all_vehicle_states()
                ep_speeds.append(ego.speed)
                # A lane change is counted whenever the ego's lane index
                # differs from the previous step.
                if ego.lane != prev_lane:
                    ep_lane_changes += 1
                steps += 1

            # Accumulate per-episode results.
            total_rewards.append(ep_reward)
            speeds.extend(ep_speeds)
            # The simulator reports collision/timeout flags in the info dict
            # returned on the terminal step.
            if info.get("collision", False):
                collisions += 1
            # An episode that ends via timeout (rather than collision) is
            # considered a "completion" -- the agent survived the full horizon.
            if info.get("timeout", False):
                completions += 1
            lengths.append(steps)
            lane_changes_all.append(ep_lane_changes)

        # Aggregate into a single metrics dataclass.
        # max(..., 1) guards against division by zero when num_episodes is 0.
        return EvalMetrics(
            avg_reward=float(np.mean(total_rewards)),
            std_reward=float(np.std(total_rewards)),
            avg_speed=float(np.mean(speeds)),
            collision_rate=collisions / max(self.num_episodes, 1),
            avg_episode_length=float(np.mean(lengths)),
            lane_changes_per_episode=float(np.mean(lane_changes_all)),
            completion_rate=completions / max(self.num_episodes, 1),
        )

    def compare(self, agents: Dict[str, Agent]) -> Dict[str, EvalMetrics]:
        """Evaluate multiple agents on the same simulator and return results.

        Args:
            agents: A mapping from human-readable agent name to Agent instance.

        Returns:
            A dictionary mapping agent name to :class:`EvalMetrics`.
        """
        results = {}
        for name, agent in agents.items():
            results[name] = self.evaluate(agent)
        return results

    @staticmethod
    def print_comparison(results: Dict[str, EvalMetrics]):
        """Pretty-print a comparison table of evaluation results.

        Produces a fixed-width table to stdout with one row per agent and
        columns for all tracked metrics.

        Args:
            results: Dictionary returned by :meth:`compare`.
        """
        header = (f"{'Agent':<22} {'AvgRew':>8} {'StdRew':>8} "
                  f"{'AvgSpd':>8} {'ColRate':>8} {'AvgLen':>8} "
                  f"{'LnChg':>8} {'Compl':>8}")
        print(header)
        print("-" * len(header))
        for name, m in results.items():
            print(f"{name:<22} {m.avg_reward:>8.1f} {m.std_reward:>8.1f} "
                  f"{m.avg_speed:>8.1f} {m.collision_rate:>8.2f} "
                  f"{m.avg_episode_length:>8.1f} "
                  f"{m.lane_changes_per_episode:>8.1f} "
                  f"{m.completion_rate:>8.2f}")
