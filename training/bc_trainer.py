"""Supervised training loop for behaviour cloning."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.policy_net import PolicyNetwork


class ObsNormalizer:
    """Simple mean/std normalizer fitted on training data."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.clip(std, 1e-6, None).astype(np.float32)

    @staticmethod
    def fit(obs: np.ndarray) -> "ObsNormalizer":
        return ObsNormalizer(obs.mean(axis=0), obs.std(axis=0))

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return ((obs - self.mean) / self.std).astype(np.float32)


class BCTrainer:
    """Train a PolicyNetwork via supervised learning on expert demonstrations.

    Uses inverse-frequency class weighting so that rare but critical
    actions (like braking) get proportionally more weight in the loss.
    """

    def __init__(self, policy: PolicyNetwork, lr: float = 3e-4,
                 batch_size: int = 64):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.batch_size = batch_size
        self.device = next(policy.parameters()).device

    def _compute_class_weights(self, actions: np.ndarray,
                                num_classes: int) -> torch.Tensor:
        """Compute inverse-frequency weights for each action class.

        Actions that appear rarely get higher weight so the network
        learns them despite class imbalance.
        """
        counts = np.bincount(actions, minlength=num_classes).astype(np.float32)
        # Avoid division by zero for unseen actions
        counts[counts == 0] = 1.0
        weights = len(actions) / (num_classes * counts)
        return torch.FloatTensor(weights).to(self.device)

    def train(self, obs: np.ndarray, actions: np.ndarray,
              num_epochs: int = 50, verbose: bool = True) -> tuple:
        """Train on collected (obs, action) pairs.

        Args:
            obs: (N, obs_dim) float array
            actions: (N,) int array of action indices
            num_epochs: number of passes over the data
            verbose: print loss every 10 epochs

        Returns:
            Tuple of (losses list, ObsNormalizer).
        """
        # Fit normalizer on training data
        self.normalizer = ObsNormalizer.fit(obs)
        obs_norm = self.normalizer.normalize(obs)

        num_classes = self.policy.num_actions
        weights = self._compute_class_weights(actions, num_classes)
        criterion = nn.CrossEntropyLoss(weight=weights)

        if verbose:
            unique, counts = np.unique(actions, return_counts=True)
            print("  Class weights:")
            from utils.types import Action
            for u, c in zip(unique, counts):
                a = Action.from_index(u)
                w = weights[u].item()
                print(f"    {a.longitudinal.name:>12}/{a.lateral.name:<6} "
                      f"count={c:>5}  weight={w:.2f}")

        dataset = TensorDataset(
            torch.FloatTensor(obs_norm).to(self.device),
            torch.LongTensor(actions).to(self.device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        losses = []

        self.policy.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for obs_batch, act_batch in loader:
                logits = self.policy(obs_batch)
                loss = criterion(logits, act_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"BC Epoch {epoch+1}/{num_epochs}  loss={avg_loss:.4f}")

        return losses, self.normalizer
