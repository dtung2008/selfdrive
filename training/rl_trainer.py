"""REINFORCE policy gradient trainer with entropy bonus and normalization."""
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from models.policy_net import PolicyNetwork
from environment.simulator import Simulator
from utils.types import Action


class RunningNormalizer:
    """Online mean/std tracker for observation normalization."""

    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        """Update running stats with a single observation."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var + 1e-8)
        return ((x - self.mean) / std).astype(np.float32)


class RLTrainer:
    """Train a PolicyNetwork via REINFORCE with:
      - Running observation normalization
      - Per-step return normalization (advantage whitening)
      - Entropy bonus to encourage exploration
      - Gradient clipping
    """

    def __init__(self, policy: PolicyNetwork, simulator: Simulator,
                 lr: float = 1e-3, gamma: float = 0.99,
                 entropy_coef: float = 0.02):
        self.policy = policy
        self.sim = simulator
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = next(policy.parameters()).device
        self.baseline = 0.0
        self.obs_normalizer = None  # can be set externally (e.g. from BC)

    def train(self, num_episodes: int = 500,
              verbose: bool = True) -> list:
        """Train for given number of episodes.

        Returns:
            List of per-episode total rewards.
        """
        episode_rewards = []

        for ep in range(num_episodes):
            obs = self.sim.reset()

            # Init normalizer on first episode
            if self.obs_normalizer is None:
                self.obs_normalizer = RunningNormalizer(len(obs))

            done = False
            log_probs = []
            entropies = []
            rewards = []

            self.policy.train()
            while not done:
                if hasattr(self.obs_normalizer, 'update'):
                    self.obs_normalizer.update(obs)
                obs_norm = self.obs_normalizer.normalize(obs)
                obs_t = torch.FloatTensor(obs_norm).unsqueeze(0).to(self.device)

                logits = self.policy(obs_t)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)
                entropy = dist.entropy()

                log_probs.append(log_prob)
                entropies.append(entropy)

                action = Action.from_index(action_idx.item())
                next_obs, reward, done, info = self.sim.step(action)
                rewards.append(reward)
                obs = next_obs

            # Compute discounted returns
            returns = self._compute_returns(rewards)
            total_reward = sum(rewards)
            episode_rewards.append(total_reward)

            # Update baseline
            self.baseline = 0.95 * self.baseline + 0.05 * total_reward

            # Normalize advantages (return - baseline) for stability
            returns_t = torch.FloatTensor(returns).to(self.device)
            advantages = returns_t - self.baseline
            if len(advantages) > 1:
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / adv_std

            # Policy gradient loss + entropy bonus
            policy_loss = 0.0
            entropy_bonus = 0.0
            for lp, adv, ent in zip(log_probs, advantages, entropies):
                policy_loss -= lp * adv.detach()
                entropy_bonus -= ent

            loss = policy_loss + self.entropy_coef * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            if verbose and (ep + 1) % 50 == 0:
                recent = np.mean(episode_rewards[-50:])
                avg_ent = np.mean([e.item() for e in entropies])
                print(f"RL Episode {ep+1}/{num_episodes}  "
                      f"avg_reward(50)={recent:.2f}  entropy={avg_ent:.3f}")

        return episode_rewards

    def _compute_returns(self, rewards):
        """Compute discounted returns for each timestep."""
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns
