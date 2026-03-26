"""Evaluate agents and produce comparison metrics."""
from typing import Dict, List
from dataclasses import dataclass, field
import numpy as np
from agents.base import Agent
from environment.simulator import Simulator
from utils.types import Action, LateralAction


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics for one agent."""
    avg_reward: float = 0.0
    std_reward: float = 0.0
    avg_speed: float = 0.0
    collision_rate: float = 0.0
    avg_episode_length: float = 0.0
    lane_changes_per_episode: float = 0.0
    completion_rate: float = 0.0  # fraction of episodes reaching timeout


class Evaluator:
    """Run an agent for N episodes and compute metrics."""

    def __init__(self, simulator: Simulator, num_episodes: int = 20):
        self.sim = simulator
        self.num_episodes = num_episodes

    def evaluate(self, agent: Agent) -> EvalMetrics:
        """Evaluate an agent over multiple episodes."""
        total_rewards = []
        speeds = []
        collisions = 0
        lengths = []
        lane_changes_all = []
        completions = 0

        for _ in range(self.num_episodes):
            obs = self.sim.reset()
            agent.reset()
            done = False
            ep_reward = 0.0
            ep_speeds = []
            ep_lane_changes = 0
            steps = 0

            while not done:
                prev_lane = self.sim.ego.lane
                action = agent.act(obs)
                obs, reward, done, info = self.sim.step(action)
                ep_reward += reward
                ego, _ = self.sim.get_all_vehicle_states()
                ep_speeds.append(ego.speed)
                if ego.lane != prev_lane:
                    ep_lane_changes += 1
                steps += 1

            total_rewards.append(ep_reward)
            speeds.extend(ep_speeds)
            if info.get("collision", False):
                collisions += 1
            if info.get("timeout", False):
                completions += 1
            lengths.append(steps)
            lane_changes_all.append(ep_lane_changes)

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
        """Evaluate multiple agents and return metrics dict."""
        results = {}
        for name, agent in agents.items():
            results[name] = self.evaluate(agent)
        return results

    @staticmethod
    def print_comparison(results: Dict[str, EvalMetrics]):
        """Pretty-print comparison table."""
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
