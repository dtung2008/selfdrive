"""Collect trajectory data by running an agent in the simulator."""
from typing import List
from agents.base import Agent
from environment.simulator import Simulator
from utils.types import Trajectory


class DataCollector:
    """Run an agent in the simulator and collect (obs, action, reward, next_obs) data."""

    def __init__(self, simulator: Simulator, agent: Agent):
        self.sim = simulator
        self.agent = agent

    def collect_episode(self) -> Trajectory:
        """Run one full episode, return Trajectory."""
        traj = Trajectory()
        obs = self.sim.reset()
        self.agent.reset()
        done = False

        while not done:
            action = self.agent.act(obs)
            action_idx = action.to_index()
            next_obs, reward, done, info = self.sim.step(action)
            traj.add(obs, action_idx, reward, next_obs, done)
            obs = next_obs

        return traj

    def collect_episodes(self, num_episodes: int) -> List[Trajectory]:
        """Collect multiple episodes."""
        trajectories = []
        for _ in range(num_episodes):
            traj = self.collect_episode()
            trajectories.append(traj)
        return trajectories


def trajectories_to_arrays(trajectories: List[Trajectory]):
    """Convert list of trajectories to flat numpy arrays for training.

    Returns:
        obs_all: (N, obs_dim)
        actions_all: (N,)
        rewards_all: (N,)
        next_obs_all: (N, obs_dim)
        dones_all: (N,)
    """
    import numpy as np
    obs_list, act_list, rew_list, nobs_list, done_list = [], [], [], [], []
    for traj in trajectories:
        for t in traj.transitions:
            obs_list.append(t.obs)
            act_list.append(t.action_idx)
            rew_list.append(t.reward)
            nobs_list.append(t.next_obs)
            done_list.append(float(t.done))

    return (np.array(obs_list, dtype=np.float32),
            np.array(act_list, dtype=np.int64),
            np.array(rew_list, dtype=np.float32),
            np.array(nobs_list, dtype=np.float32),
            np.array(done_list, dtype=np.float32))
