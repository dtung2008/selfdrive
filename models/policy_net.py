"""Policy network: MLP that maps observations to action probabilities."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.types import Action


class PolicyNetwork(nn.Module):
    """Simple MLP policy: obs -> action logits (9 discrete actions)."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, num_actions: int = 9):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_actions))
        self.net = nn.Sequential(*layers)
        self.num_actions = num_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action logits. Shape: (batch, num_actions)."""
        return self.net(obs)

    def get_action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action probabilities."""
        logits = self.forward(obs)
        return F.softmax(logits, dim=-1)

    def sample_action(self, obs: torch.Tensor) -> int:
        """Sample an action index from the policy."""
        with torch.no_grad():
            probs = self.get_action_probs(obs)
            dist = torch.distributions.Categorical(probs)
            return dist.sample().item()

    def greedy_action(self, obs: torch.Tensor) -> int:
        """Return the most probable action index."""
        with torch.no_grad():
            logits = self.forward(obs)
            return logits.argmax(dim=-1).item()
