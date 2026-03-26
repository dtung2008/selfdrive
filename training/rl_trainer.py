"""REINFORCE policy-gradient trainer with entropy bonus and normalisation.

This module implements the classic REINFORCE algorithm (Williams, 1992) for
training a discrete-action policy network via on-policy rollouts in the
driving simulator.  Several variance-reduction and stability techniques are
applied:

1. **Running observation normalisation** -- an online (Welford) estimator
   tracks per-feature mean and variance so that the policy always sees
   zero-mean, unit-variance inputs regardless of how the simulator's raw
   observation scale evolves.

2. **Exponential-moving-average baseline** -- a scalar baseline tracks the
   expected return and is subtracted from each episode's discounted returns
   to reduce gradient variance.

3. **Advantage whitening** -- per-episode advantages (return - baseline) are
   further standardised (zero-mean, unit-variance) to keep the gradient
   magnitude stable across episodes with very different return scales.

4. **Entropy bonus** -- a small negative-entropy penalty is added to the loss
   to discourage the policy from collapsing to a deterministic distribution
   too early, promoting exploration.

5. **Gradient clipping** -- gradients are clipped by global norm (0.5) to
   prevent destructive parameter updates from high-variance episodes.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from models.policy_net import PolicyNetwork
from environment.simulator import Simulator
from utils.types import Action


class RunningNormalizer:
    """Online (Welford) mean/variance tracker for observation normalisation.

    Uses an incremental update rule to maintain running mean and variance
    without storing all past observations.  This is essential for RL where
    data arrives one observation at a time during rollouts.

    The update follows Welford's online algorithm:
        delta   = x - mean
        mean   += delta / count
        delta2  = x - mean            (note: uses *updated* mean)
        var    += (delta * delta2 - var) / count
    """

    def __init__(self, dim: int):
        """Initialise the normaliser for a given observation dimensionality.

        Args:
            dim: Number of features in the observation vector.
        """
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)   # Start with unit variance.
        # Small positive count to avoid division by zero on the first update.
        self.count = 1e-4

    def update(self, x: np.ndarray):
        """Incorporate a single observation into the running statistics.

        Args:
            x: A single observation vector of shape (dim,).
        """
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        # Second delta uses the *updated* mean (Welford's trick).
        delta2 = x - self.mean
        # Incremental variance update.
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalise an observation using the running statistics.

        Args:
            x: Raw observation, shape (dim,) or (N, dim).

        Returns:
            Normalised observation as float32.
        """
        # Add a small epsilon (1e-8) inside the sqrt to prevent NaN when
        # variance is exactly zero (e.g., constant feature early in training).
        std = np.sqrt(self.var + 1e-8)
        return ((x - self.mean) / std).astype(np.float32)


class RLTrainer:
    """Train a PolicyNetwork via REINFORCE with:

    - Running observation normalisation
    - Per-step return normalisation (advantage whitening)
    - Entropy bonus to encourage exploration
    - Gradient clipping

    Typical workflow::

        trainer = RLTrainer(policy, simulator)
        # Optionally warm-start from a BC normaliser:
        trainer.obs_normalizer = bc_normalizer
        rewards = trainer.train(num_episodes=500)
    """

    def __init__(self, policy: PolicyNetwork, simulator: Simulator,
                 lr: float = 1e-3, gamma: float = 0.99,
                 entropy_coef: float = 0.02):
        """Initialise the RL trainer.

        Args:
            policy:       The policy network to train.
            simulator:    The driving-environment simulator for rollouts.
            lr:           Learning rate for the Adam optimiser.
            gamma:        Discount factor for computing returns.  Values close
                          to 1.0 make the agent more far-sighted.
            entropy_coef: Coefficient for the entropy bonus term.  Higher
                          values encourage more exploration at the cost of
                          slower convergence.
        """
        self.policy = policy
        self.sim = simulator
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = next(policy.parameters()).device
        # Exponential-moving-average baseline for variance reduction.
        self.baseline = 0.0
        # Observation normaliser; can be injected externally (e.g., from a
        # pre-trained BC normaliser) or will be created on the first episode.
        self.obs_normalizer = None

    def train(self, num_episodes: int = 500,
              verbose: bool = True) -> list:
        """Run the REINFORCE training loop for a given number of episodes.

        Each episode performs a full rollout (reset -> act -> step until done),
        collects log-probabilities, entropies, and rewards at each step, then
        computes the policy-gradient loss and updates the network once per
        episode.

        Args:
            num_episodes: Total number of on-policy rollout episodes.
            verbose:      If True, print rolling-average reward and entropy
                          every 50 episodes.

        Returns:
            A list of per-episode total (undiscounted) rewards, useful for
            plotting learning curves.
        """
        episode_rewards = []

        for ep in range(num_episodes):
            obs = self.sim.reset()

            # Lazily initialise the observation normaliser on the first
            # episode when no external normaliser has been provided.
            if self.obs_normalizer is None:
                self.obs_normalizer = RunningNormalizer(len(obs))

            done = False
            log_probs = []   # Log-probability of the sampled action at each step.
            entropies = []   # Categorical entropy of the policy at each step.
            rewards = []     # Raw reward at each step.

            # ------------------------------------------------------------------
            # Rollout: collect one full episode.
            # ------------------------------------------------------------------
            self.policy.train()
            while not done:
                # Update running observation statistics (only for the online
                # RunningNormalizer; the BC ObsNormalizer has no update method).
                if hasattr(self.obs_normalizer, 'update'):
                    self.obs_normalizer.update(obs)
                # Normalise the observation before feeding it to the network.
                obs_norm = self.obs_normalizer.normalize(obs)
                # Convert to a batch-of-one tensor for the policy forward pass.
                obs_t = torch.FloatTensor(obs_norm).unsqueeze(0).to(self.device)

                # Forward pass: obtain raw logits from the policy network.
                logits = self.policy(obs_t)
                # Convert logits to a proper probability distribution.
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                # Sample an action from the policy distribution.
                action_idx = dist.sample()
                # Record the log-probability (needed for the policy gradient)
                # and entropy (needed for the exploration bonus).
                log_prob = dist.log_prob(action_idx)
                entropy = dist.entropy()

                log_probs.append(log_prob)
                entropies.append(entropy)

                # Convert the integer action index back to a structured Action
                # object for the simulator.
                action = Action.from_index(action_idx.item())
                next_obs, reward, done, info = self.sim.step(action)
                rewards.append(reward)
                obs = next_obs

            # ------------------------------------------------------------------
            # Post-rollout: compute returns, advantages, and update the policy.
            # ------------------------------------------------------------------

            # Compute discounted returns G_t for each timestep.
            returns = self._compute_returns(rewards)
            total_reward = sum(rewards)
            episode_rewards.append(total_reward)

            # Update the exponential-moving-average baseline.
            # The smoothing factor (0.95) controls how quickly the baseline
            # adapts: values near 1.0 make it more stable but slower to track
            # non-stationarity in the reward distribution.
            self.baseline = 0.95 * self.baseline + 0.05 * total_reward

            # ------------------------------------------------------------------
            # Advantage whitening: normalise (return - baseline) to have zero
            # mean and unit variance.  This keeps gradient magnitudes stable
            # across episodes with widely varying return scales.
            # ------------------------------------------------------------------
            returns_t = torch.FloatTensor(returns).to(self.device)
            advantages = returns_t - self.baseline
            if len(advantages) > 1:
                adv_std = advantages.std()
                # Only normalise if standard deviation is non-trivial to avoid
                # dividing by near-zero and amplifying noise.
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / adv_std

            # ------------------------------------------------------------------
            # Compute the REINFORCE loss + entropy bonus.
            #
            # policy_loss = -sum_t [ log pi(a_t|s_t) * A_t ]
            #   Minimising this pushes up the probability of actions with
            #   positive advantage and pushes down actions with negative
            #   advantage.
            #
            # entropy_bonus = -sum_t [ H(pi(.|s_t)) ]
            #   Subtracting entropy from the loss *increases* it when entropy
            #   is low, penalising overly deterministic policies.
            # ------------------------------------------------------------------
            policy_loss = 0.0
            entropy_bonus = 0.0
            for lp, adv, ent in zip(log_probs, advantages, entropies):
                # Detach advantage so gradients only flow through log_prob.
                policy_loss -= lp * adv.detach()
                # Negative sign: we *subtract* entropy from the reward
                # (equivalently, add negative entropy to the loss), so that
                # higher entropy is encouraged.
                entropy_bonus -= ent

            # Combined loss: policy gradient + weighted entropy penalty.
            loss = policy_loss + self.entropy_coef * entropy_bonus

            # Backpropagation and parameter update with gradient clipping.
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients by global L2 norm to prevent destructive updates
            # from high-variance episodes.
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            # Periodic logging.
            if verbose and (ep + 1) % 50 == 0:
                recent = np.mean(episode_rewards[-50:])
                avg_ent = np.mean([e.item() for e in entropies])
                print(f"RL Episode {ep+1}/{num_episodes}  "
                      f"avg_reward(50)={recent:.2f}  entropy={avg_ent:.3f}")

        return episode_rewards

    def _compute_returns(self, rewards):
        """Compute discounted returns G_t for each timestep.

        Uses the standard backward recursion:
            G_T     = r_T
            G_t     = r_t + gamma * G_{t+1}

        This traverses the reward list in reverse, accumulating the
        discounted sum, then reverses the result so that returns[t]
        corresponds to the return starting from timestep t.

        Args:
            rewards: List of per-step scalar rewards [r_0, r_1, ..., r_T].

        Returns:
            A list of discounted returns [G_0, G_1, ..., G_T] with the same
            length as *rewards*.
        """
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            # Prepend so the final list is in chronological order.
            returns.insert(0, G)
        return returns
