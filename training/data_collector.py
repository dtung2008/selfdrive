"""Collect trajectory data by running an agent in the simulator.

This module provides utilities for gathering experience data from the driving
simulator.  A :class:`DataCollector` pairs an :class:`Agent` with a
:class:`Simulator`, runs one or more episodes, and returns the experience as
:class:`Trajectory` objects.  The helper function :func:`trajectories_to_arrays`
flattens these trajectories into contiguous NumPy arrays suitable for training
supervised or reinforcement-learning models.

Data collection strategy
------------------------
Each episode runs the standard environment loop (reset -> act -> step) until
the simulator signals ``done`` (collision, timeout, or other termination).  The
agent is also reset at the start of each episode so that any internal state
(e.g., recurrent hidden states) does not leak across episodes.

The resulting transitions are stored in order, which preserves temporal
structure for algorithms that need sequential data (e.g., world-model
training).  For algorithms that treat data as i.i.d. (behaviour cloning), the
DataLoader in the trainer will shuffle samples.
"""
from typing import List
from agents.base import Agent
from environment.simulator import Simulator
from utils.types import Trajectory


class DataCollector:
    """Run an agent in the simulator and collect (obs, action, reward, next_obs) data.

    This class is stateless between calls -- it does not accumulate data across
    ``collect_episode`` invocations.  Callers are responsible for aggregating
    trajectories externally.
    """

    def __init__(self, simulator: Simulator, agent: Agent):
        """Initialise the collector with a simulator and an agent.

        Args:
            simulator: The driving-environment simulator instance.
            agent:     The agent whose behaviour will be recorded.  For
                       behaviour cloning this is typically an expert (e.g.,
                       a rule-based driver); for RL evaluation it can be
                       the learning agent itself.
        """
        self.sim = simulator
        self.agent = agent

    def collect_episode(self) -> Trajectory:
        """Run one full episode from reset to termination.

        The simulator and agent are both reset before the episode begins.
        At each timestep the agent selects an action, the simulator advances,
        and the resulting transition (obs, action_idx, reward, next_obs, done)
        is appended to the trajectory.

        Returns:
            A :class:`Trajectory` containing all transitions for the episode.
        """
        traj = Trajectory()
        obs = self.sim.reset()       # Reset simulator and obtain initial observation.
        self.agent.reset()            # Reset agent's internal state (if any).
        done = False

        while not done:
            # Agent selects an Action (a composite of longitudinal + lateral).
            action = self.agent.act(obs)
            # Convert the structured Action to a flat integer index for storage.
            action_idx = action.to_index()
            # Step the simulator forward and collect the transition tuple.
            next_obs, reward, done, info = self.sim.step(action)
            # Record this transition in the trajectory.
            traj.add(obs, action_idx, reward, next_obs, done)
            # Advance to the next observation.
            obs = next_obs

        return traj

    def collect_episodes(self, num_episodes: int) -> List[Trajectory]:
        """Collect multiple episodes sequentially.

        Args:
            num_episodes: Number of episodes to run.

        Returns:
            A list of :class:`Trajectory` objects, one per episode.
        """
        trajectories = []
        for _ in range(num_episodes):
            traj = self.collect_episode()
            trajectories.append(traj)
        return trajectories


def trajectories_to_arrays(trajectories: List[Trajectory]):
    """Convert a list of trajectories to flat NumPy arrays for training.

    All transitions from every trajectory are concatenated into single arrays.
    This discards episode boundaries, which is fine for algorithms that treat
    each transition independently (e.g., behaviour cloning, world-model
    training).  For algorithms that need episode structure, operate on the
    :class:`Trajectory` objects directly.

    Args:
        trajectories: A list of :class:`Trajectory` objects produced by
                      :class:`DataCollector`.

    Returns:
        A 5-tuple of NumPy arrays::

            obs_all      -- (N, obs_dim)  float32  observations
            actions_all  -- (N,)          int64    action indices
            rewards_all  -- (N,)          float32  scalar rewards
            next_obs_all -- (N, obs_dim)  float32  next observations
            dones_all    -- (N,)          float32  done flags (0.0 or 1.0)
    """
    import numpy as np

    # Accumulate individual fields from every transition across all episodes.
    obs_list, act_list, rew_list, nobs_list, done_list = [], [], [], [], []
    for traj in trajectories:
        for t in traj.transitions:
            obs_list.append(t.obs)
            act_list.append(t.action_idx)
            rew_list.append(t.reward)
            nobs_list.append(t.next_obs)
            # Convert boolean done flag to float for downstream tensor ops.
            done_list.append(float(t.done))

    # Stack into contiguous arrays with explicit dtypes for PyTorch compatibility.
    return (np.array(obs_list, dtype=np.float32),
            np.array(act_list, dtype=np.int64),
            np.array(rew_list, dtype=np.float32),
            np.array(nobs_list, dtype=np.float32),
            np.array(done_list, dtype=np.float32))
