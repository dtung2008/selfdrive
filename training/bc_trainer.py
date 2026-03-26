"""Supervised training loop for behaviour cloning.

Behaviour cloning (BC) is an imitation-learning approach where a neural network
policy is trained via supervised learning to mimic expert demonstrations.  Given
a dataset of (observation, action) pairs collected from an expert driver, the
policy learns to map observations to action probabilities by minimising the
cross-entropy loss between predicted and expert actions.

Key design choices in this module:
  - Observations are z-score normalised (zero-mean, unit-variance) before
    training so that features with different scales contribute equally.
  - Inverse-frequency class weighting is applied to the cross-entropy loss so
    that rare but safety-critical actions (e.g., hard braking) are not drowned
    out by dominant actions (e.g., maintain speed).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.policy_net import PolicyNetwork


class ObsNormalizer:
    """Simple per-feature mean/std normalizer fitted on training data.

    Stores the mean and standard deviation computed from a training set and
    applies the standard z-score transformation:  x' = (x - mean) / std.
    The normalizer must be persisted alongside the trained policy so that
    observations at inference time undergo the same transformation.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        """Initialise with pre-computed statistics.

        Args:
            mean: Per-feature mean, shape (obs_dim,).
            std:  Per-feature standard deviation, shape (obs_dim,).  Values
                  below 1e-6 are clipped to avoid division-by-zero for
                  constant features (e.g., padding zeros).
        """
        self.mean = mean.astype(np.float32)
        # Clip std to a small positive floor to prevent division by zero
        # when a feature has zero variance (e.g., always-zero padding).
        self.std = np.clip(std, 1e-6, None).astype(np.float32)

    @staticmethod
    def fit(obs: np.ndarray) -> "ObsNormalizer":
        """Compute mean/std from data and return a new ObsNormalizer.

        Args:
            obs: Observation matrix of shape (N, obs_dim).

        Returns:
            A fitted ObsNormalizer instance ready for normalisation.
        """
        return ObsNormalizer(obs.mean(axis=0), obs.std(axis=0))

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Apply z-score normalisation: (obs - mean) / std.

        Args:
            obs: Raw observations, shape (N, obs_dim) or (obs_dim,).

        Returns:
            Normalised observations as float32 with the same shape.
        """
        return ((obs - self.mean) / self.std).astype(np.float32)


class BCTrainer:
    """Train a PolicyNetwork via supervised learning on expert demonstrations.

    Uses inverse-frequency class weighting so that rare but critical
    actions (like braking) get proportionally more weight in the loss.

    Typical workflow::

        trainer = BCTrainer(policy)
        losses, normalizer = trainer.train(obs, actions)
        # Save `policy` and `normalizer` for inference.
    """

    def __init__(self, policy: PolicyNetwork, lr: float = 3e-4,
                 batch_size: int = 64):
        """Initialise the behaviour-cloning trainer.

        Args:
            policy:     The policy network to train.  Its device is detected
                        automatically from its parameters.
            lr:         Learning rate for Adam optimiser (default 3e-4).
            batch_size: Mini-batch size for stochastic gradient descent.
        """
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.batch_size = batch_size
        # Detect device (CPU / CUDA) from the policy's existing parameters.
        self.device = next(policy.parameters()).device

    def _compute_class_weights(self, actions: np.ndarray,
                                num_classes: int) -> torch.Tensor:
        """Compute inverse-frequency weights for each action class.

        Actions that appear rarely get higher weight so the network
        learns them despite class imbalance.  The formula is:

            weight_c = N / (C * count_c)

        where N is the total number of samples, C is the number of classes,
        and count_c is the number of samples belonging to class c.

        Args:
            actions:     1-D integer array of action indices, shape (N,).
            num_classes: Total number of discrete action classes.

        Returns:
            A float tensor of shape (num_classes,) on the policy device,
            suitable for passing to ``nn.CrossEntropyLoss(weight=...)``.
        """
        counts = np.bincount(actions, minlength=num_classes).astype(np.float32)
        # Replace zero counts with 1.0 to avoid division by zero for action
        # classes that never appear in the training set.
        counts[counts == 0] = 1.0
        # Inverse-frequency weighting: rarer actions receive higher weight.
        weights = len(actions) / (num_classes * counts)
        return torch.FloatTensor(weights).to(self.device)

    def train(self, obs: np.ndarray, actions: np.ndarray,
              num_epochs: int = 50, verbose: bool = True) -> tuple:
        """Train on collected (obs, action) pairs.

        The method fits an ObsNormalizer on the provided observations,
        computes class weights, builds a PyTorch DataLoader, and runs
        the standard supervised training loop (forward -> loss -> backward
        -> optimiser step) for the requested number of epochs.

        Args:
            obs:        Observation matrix, shape (N, obs_dim), float.
            actions:    Action indices, shape (N,), int.
            num_epochs: Number of full passes over the dataset.
            verbose:    If True, print class-weight summary and loss every
                        10 epochs.

        Returns:
            A tuple of ``(losses, normalizer)`` where *losses* is a list of
            per-epoch average cross-entropy loss values, and *normalizer* is
            the fitted :class:`ObsNormalizer` (needed at inference time).
        """
        # ------------------------------------------------------------------
        # 1. Fit normalizer on training data and normalise observations.
        # ------------------------------------------------------------------
        self.normalizer = ObsNormalizer.fit(obs)
        obs_norm = self.normalizer.normalize(obs)

        # ------------------------------------------------------------------
        # 2. Build inverse-frequency class weights and the loss function.
        # ------------------------------------------------------------------
        num_classes = self.policy.num_actions
        weights = self._compute_class_weights(actions, num_classes)
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Optionally print a human-readable summary of the class weights so
        # the practitioner can verify that rare actions are up-weighted.
        if verbose:
            unique, counts = np.unique(actions, return_counts=True)
            print("  Class weights:")
            from utils.types import Action
            for u, c in zip(unique, counts):
                a = Action.from_index(u)
                w = weights[u].item()
                print(f"    {a.longitudinal.name:>12}/{a.lateral.name:<6} "
                      f"count={c:>5}  weight={w:.2f}")

        # ------------------------------------------------------------------
        # 3. Wrap data in a TensorDataset and DataLoader for mini-batching.
        # ------------------------------------------------------------------
        dataset = TensorDataset(
            torch.FloatTensor(obs_norm).to(self.device),
            torch.LongTensor(actions).to(self.device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        losses = []

        # ------------------------------------------------------------------
        # 4. Training loop: iterate over epochs and mini-batches.
        # ------------------------------------------------------------------
        self.policy.train()  # Switch to training mode (enables dropout, etc.)
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for obs_batch, act_batch in loader:
                # Forward pass: compute action logits from normalised obs.
                logits = self.policy(obs_batch)
                # Weighted cross-entropy loss between predicted and expert action.
                loss = criterion(logits, act_batch)
                # Standard backward pass + parameter update.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Accumulate loss for epoch-level reporting.
                epoch_loss += loss.item()
                n_batches += 1

            # Compute average loss across all batches in this epoch.
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"BC Epoch {epoch+1}/{num_epochs}  loss={avg_loss:.4f}")

        return losses, self.normalizer
