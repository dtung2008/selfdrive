"""Abstract base class for all driving agents.

This module defines the ``Agent`` interface that every concrete agent
(rule-based experts, imitation-learning policies, RL policies, etc.) must
implement.  The interface is intentionally minimal so that the training
loop, evaluation harness, and data-collection pipeline can treat every
agent identically regardless of its internal decision-making strategy.

Typical usage::

    agent = SomeConcretAgent(...)
    agent.reset()                     # start of a new episode
    action = agent.act(observation)   # called every simulation step
"""

from abc import ABC, abstractmethod
import numpy as np
from utils.types import Action


class Agent(ABC):
    """Base agent interface that all driving agents must implement.

    Subclasses are required to override :meth:`act`.  Optionally they may
    override :meth:`reset` to clear any internal episode state (e.g.
    recurrent hidden states, step counters, or lane-change cooldowns).
    """

    @abstractmethod
    def act(self, obs: np.ndarray) -> Action:
        """Select an action for the current simulation step.

        Args:
            obs: A 1-D NumPy array whose layout is defined by
                :class:`utils.config.ObservationConfig`.  The first few
                elements encode ego-vehicle state (speed, lane, x-position)
                and the remaining elements encode the k nearest neighbors.

        Returns:
            An :class:`Action` comprising a longitudinal component
            (accelerate / decelerate / keep) and a lateral component
            (left / right / keep lane).
        """
        ...

    def reset(self):
        """Reset any internal state at the start of a new episode.

        The default implementation is a no-op.  Subclasses that maintain
        episode-level bookkeeping (e.g. step counters, cooldown timers,
        RNN hidden states) should override this method.
        """
        pass
