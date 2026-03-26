"""Training loop for the attention-based world model.

A world model learns the environment's dynamics -- given the current
observation and an action, it predicts the next observation.  This enables
*model-based* planning: instead of interacting with the real simulator, an
agent can "imagine" trajectories by repeatedly querying the world model, then
choose the action sequence with the best predicted outcome.

Training strategy
-----------------
Rather than predicting raw next-observation values, the model is trained to
predict **deltas in normalised space**:

    target = normalise(next_obs) - normalise(obs)

Predicting small residuals is easier to learn than predicting absolute states,
because the residuals have lower magnitude and less feature-to-feature scale
variation.  The normaliser's ``inverse_transform`` can then recover absolute
values at inference time.

The loss is mean-squared error (MSE) between predicted and true normalised
next-observations.  Gradient clipping (global L2 norm <= 1.0) is applied to
stabilise training.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.world_model import AttentionWorldModel


class ObsNormalizer:
    """Per-feature zero-mean unit-variance normalizer.

    Stores mean and standard deviation so that the same normalisation can be
    applied at inference time (e.g., inside the learned-model planner).

    Unlike the BC trainer's ``ObsNormalizer``, this version exposes a
    scikit-learn-style ``fit`` / ``transform`` / ``inverse_transform`` API
    so that predictions can be mapped back to the original observation space.
    """

    def __init__(self):
        """Initialise with empty statistics (call :meth:`fit` before use)."""
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        """Compute per-feature mean and standard deviation from data.

        Args:
            data: Observation matrix of shape (N, obs_dim).

        Returns:
            ``self``, to allow method chaining (e.g., ``norm.fit(x).transform(x)``).
        """
        self.mean = data.mean(axis=0).astype(np.float32)
        self.std = data.std(axis=0).astype(np.float32)
        # Replace near-zero standard deviations with 1.0 to avoid division
        # by zero for constant features (e.g., padding zeros or features
        # that are identical across all training samples).
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply z-score normalisation: (data - mean) / std.

        Args:
            data: Observations, shape (N, obs_dim) or (obs_dim,).

        Returns:
            Normalised observations as float32.
        """
        return ((data - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the normalisation: data * std + mean.

        Used to map model predictions (in normalised space) back to the
        simulator's original observation scale.

        Args:
            data: Normalised observations, shape (N, obs_dim) or (obs_dim,).

        Returns:
            De-normalised observations as float32.
        """
        return (data * self.std + self.mean).astype(np.float32)


class WorldModelTrainer:
    """Train an AttentionWorldModel to predict next observation from (obs, action).

    Trains on *delta prediction in normalised space*:
        target = normalise(next_obs) - normalise(obs)
    This means the model only needs to predict small residuals, which is
    much easier to learn than predicting absolute next states.

    Note: although the docstring and design intent describe delta prediction,
    the current implementation trains against the full normalised next_obs
    (``nobs_norm``) rather than the delta.  This still benefits from
    normalisation but does not exploit the residual structure.  A future
    improvement could switch the target to ``nobs_norm - obs_norm``.
    """

    def __init__(self, world_model: AttentionWorldModel,
                 lr: float = 3e-4, batch_size: int = 64):
        """Initialise the world-model trainer.

        Args:
            world_model: The attention-based world model to train.
            lr:          Learning rate for the Adam optimiser.
            batch_size:  Mini-batch size for stochastic gradient descent.
        """
        self.wm = world_model
        self.optimizer = optim.Adam(world_model.parameters(), lr=lr)
        # MSE loss between predicted and true normalised next-observations.
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.device = next(world_model.parameters()).device
        self.normalizer = ObsNormalizer()

    def train(self, obs: np.ndarray, actions: np.ndarray,
              next_obs: np.ndarray, num_epochs: int = 50,
              verbose: bool = True) -> list:
        """Train on (obs, action, next_obs) tuples.

        The normaliser is fitted on the union of ``obs`` and ``next_obs`` so
        that both current and next observations share the same scale.  Both
        are then normalised, and the model is trained to predict normalised
        next-observations given normalised current observations and action
        indices.

        Args:
            obs:        Current observations, shape (N, obs_dim), float.
            actions:    Action indices, shape (N,), int.
            next_obs:   Next observations, shape (N, obs_dim), float.
            num_epochs: Number of full passes over the dataset.
            verbose:    If True, print loss every 10 epochs.

        Returns:
            A list of per-epoch average MSE losses.
        """
        # ------------------------------------------------------------------
        # 1. Fit the normaliser on the concatenation of obs and next_obs.
        #    Using the union ensures both share the same mean/std, which is
        #    important because the model's output should live in the same
        #    normalised space as its input.
        # ------------------------------------------------------------------
        all_obs = np.concatenate([obs, next_obs], axis=0)
        self.normalizer.fit(all_obs)

        obs_norm = self.normalizer.transform(obs)
        nobs_norm = self.normalizer.transform(next_obs)

        # ------------------------------------------------------------------
        # 2. Build PyTorch dataset and data loader for mini-batch training.
        # ------------------------------------------------------------------
        dataset = TensorDataset(
            torch.FloatTensor(obs_norm).to(self.device),
            torch.LongTensor(actions).to(self.device),
            torch.FloatTensor(nobs_norm).to(self.device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        losses = []

        # ------------------------------------------------------------------
        # 3. Training loop: iterate over epochs and mini-batches.
        # ------------------------------------------------------------------
        self.wm.train()  # Enable training mode (dropout, batch-norm, etc.).
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for obs_b, act_b, nobs_b in loader:
                # Forward pass: world model predicts normalised next-obs.
                pred = self.wm(obs_b, act_b)
                # MSE loss between predicted and true normalised next-obs.
                loss = self.criterion(pred, nobs_b)
                # Backward pass and optimiser step.
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients by global L2 norm to prevent large,
                # destabilising parameter updates.
                torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
                self.optimizer.step()
                # Accumulate loss for epoch-level reporting.
                epoch_loss += loss.item()
                n_batches += 1

            # Compute average loss across all mini-batches in this epoch.
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"WM Epoch {epoch+1}/{num_epochs}  loss={avg_loss:.6f}")

        return losses
