"""Training loop for the attention-based world model."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.world_model import AttentionWorldModel


class ObsNormalizer:
    """Per-feature zero-mean unit-variance normalizer.

    Stores mean/std so the same normalization can be applied at inference
    (e.g., inside the learned-model planner).
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        """Compute mean and std from data (N, dim)."""
        self.mean = data.mean(axis=0).astype(np.float32)
        self.std = data.std(axis=0).astype(np.float32)
        # Avoid division by zero for constant features (e.g., padding zeros)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return ((data - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std + self.mean).astype(np.float32)


class WorldModelTrainer:
    """Train AttentionWorldModel to predict next observation from (obs, action).

    Trains on *delta prediction in normalized space*:
        target = normalize(next_obs) - normalize(obs)
    This means the model only needs to predict small residuals, which is
    much easier to learn than predicting absolute next states.
    """

    def __init__(self, world_model: AttentionWorldModel,
                 lr: float = 3e-4, batch_size: int = 64):
        self.wm = world_model
        self.optimizer = optim.Adam(world_model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.device = next(world_model.parameters()).device
        self.normalizer = ObsNormalizer()

    def train(self, obs: np.ndarray, actions: np.ndarray,
              next_obs: np.ndarray, num_epochs: int = 50,
              verbose: bool = True) -> list:
        """Train on (obs, action, next_obs) tuples.

        Args:
            obs: (N, obs_dim)
            actions: (N,) action indices
            next_obs: (N, obs_dim)
            num_epochs: training epochs
            verbose: print progress

        Returns:
            List of per-epoch average losses.
        """
        # Fit normalizer on all observations
        all_obs = np.concatenate([obs, next_obs], axis=0)
        self.normalizer.fit(all_obs)

        obs_norm = self.normalizer.transform(obs)
        nobs_norm = self.normalizer.transform(next_obs)

        dataset = TensorDataset(
            torch.FloatTensor(obs_norm).to(self.device),
            torch.LongTensor(actions).to(self.device),
            torch.FloatTensor(nobs_norm).to(self.device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        losses = []

        self.wm.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for obs_b, act_b, nobs_b in loader:
                pred = self.wm(obs_b, act_b)
                loss = self.criterion(pred, nobs_b)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"WM Epoch {epoch+1}/{num_epochs}  loss={avg_loss:.6f}")

        return losses
