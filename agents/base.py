"""Abstract base class for all agents."""
from abc import ABC, abstractmethod
import numpy as np
from utils.types import Action


class Agent(ABC):
    """Base agent interface."""

    @abstractmethod
    def act(self, obs: np.ndarray) -> Action:
        """Choose an action given an observation vector."""
        ...

    def reset(self):
        """Called at the start of each episode (optional override)."""
        pass
