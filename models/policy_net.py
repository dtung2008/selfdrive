"""Policy network: MLP that maps observations to action probabilities.

This module implements a simple feed-forward policy network for discrete
action spaces. The network takes a flat observation vector as input and
produces logits over a fixed set of discrete actions (default 9, covering
combinations of acceleration and lane-change manoeuvres).

The architecture is a straightforward multi-layer perceptron (MLP) with
ReLU activations between hidden layers and no activation on the output
layer (raw logits). A softmax can be applied externally or via the
convenience method `get_action_probs` to convert logits to a valid
probability distribution.

Typical usage:
    policy = PolicyNetwork(obs_dim=27, hidden_dim=128, num_layers=2)
    action = policy.sample_action(obs_tensor)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.types import Action


class PolicyNetwork(nn.Module):
    """Simple MLP policy network: observation -> action logits.

    Maps a flat observation vector to unnormalised log-probabilities (logits)
    over 9 discrete driving actions. The action space typically encodes a
    3x3 grid of (acceleration x lane-change) combinations.

    Architecture:
        Input(obs_dim)
          -> [Linear(hidden_dim) -> ReLU] x num_layers
          -> Linear(num_actions)   # output logits, no activation

    Attributes:
        net (nn.Sequential): The sequential stack of linear layers and
            ReLU activations, terminated by an output projection layer.
        num_actions (int): Size of the discrete action space.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, num_actions: int = 9):
        """Initialise the policy network.

        Args:
            obs_dim: Dimensionality of the flat observation vector
                (e.g. 3 ego features + 4 features * k neighbours).
            hidden_dim: Width of every hidden layer. All hidden layers
                share the same width for simplicity.
            num_layers: Number of hidden layers (each followed by ReLU).
                The output projection is added on top of these.
            num_actions: Number of discrete actions the policy can choose
                from. Defaults to 9 (3 acceleration x 3 lane-change).
        """
        super().__init__()

        # Build hidden layers dynamically: each is Linear + ReLU.
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim  # subsequent layers read from hidden_dim

        # Final projection to action logits -- no activation here because
        # softmax / log-softmax is applied downstream (in loss or sampling).
        layers.append(nn.Linear(hidden_dim, num_actions))

        self.net = nn.Sequential(*layers)
        self.num_actions = num_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute raw action logits from an observation.

        Args:
            obs: Observation tensor of shape ``(batch, obs_dim)``.

        Returns:
            Logits tensor of shape ``(batch, num_actions)``. These are
            unnormalised scores; apply softmax to obtain probabilities.
        """
        return self.net(obs)

    def get_action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute a normalised probability distribution over actions.

        Applies softmax to the raw logits along the action dimension so
        that the output sums to 1 and can be interpreted as probabilities.

        Args:
            obs: Observation tensor of shape ``(batch, obs_dim)``.

        Returns:
            Probability tensor of shape ``(batch, num_actions)`` where
            each row sums to 1.
        """
        logits = self.forward(obs)
        return F.softmax(logits, dim=-1)

    def sample_action(self, obs: torch.Tensor) -> int:
        """Stochastically sample a single action from the policy.

        Constructs a Categorical distribution from the action probabilities
        and draws one sample. Runs under ``torch.no_grad()`` because this
        is used at inference / environment-interaction time, not training.

        Args:
            obs: Observation tensor of shape ``(1, obs_dim)`` or
                ``(obs_dim,)`` — typically a single observation.

        Returns:
            An integer action index in ``[0, num_actions)``.
        """
        with torch.no_grad():
            probs = self.get_action_probs(obs)
            # Categorical wraps the probability vector into a distribution
            # from which we can call .sample() to get a discrete index.
            dist = torch.distributions.Categorical(probs)
            return dist.sample().item()

    def greedy_action(self, obs: torch.Tensor) -> int:
        """Return the deterministic (greedy) best action.

        Selects the action with the highest logit value. This is equivalent
        to argmax over the softmax output but avoids the unnecessary
        exponentiation.

        Args:
            obs: Observation tensor of shape ``(1, obs_dim)`` or
                ``(obs_dim,)`` — typically a single observation.

        Returns:
            An integer action index in ``[0, num_actions)``.
        """
        with torch.no_grad():
            logits = self.forward(obs)
            return logits.argmax(dim=-1).item()
