"""Tests for training: data collector, BC trainer, WM trainer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from utils.config import Config
from environment.simulator import Simulator
from agents.expert import ExpertAgent
from models.policy_net import PolicyNetwork
from models.world_model import AttentionWorldModel
from training.data_collector import DataCollector, trajectories_to_arrays
from training.bc_trainer import BCTrainer
from training.world_model_trainer import WorldModelTrainer


class TestDataCollector:
    def setup_method(self):
        cfg = Config()
        cfg.traffic.num_npcs = 2
        cfg.sim.episode_steps = 30
        self.sim = Simulator(cfg, seed=42)
        self.expert = ExpertAgent(obs_config=cfg.obs,
                                   num_lanes=cfg.road.num_lanes)
        self.collector = DataCollector(self.sim, self.expert)

    def test_collect_episode(self):
        traj = self.collector.collect_episode()
        assert len(traj) > 0
        assert len(traj) <= 30

    def test_collect_episodes(self):
        trajs = self.collector.collect_episodes(3)
        assert len(trajs) == 3
        for t in trajs:
            assert len(t) > 0

    def test_trajectories_to_arrays(self):
        trajs = self.collector.collect_episodes(2)
        obs, acts, rews, nobs, dones = trajectories_to_arrays(trajs)
        total = sum(len(t) for t in trajs)
        assert obs.shape[0] == total
        assert acts.shape[0] == total
        assert obs.shape[1] == nobs.shape[1]
        assert acts.dtype == np.int64


class TestBCTrainer:
    def test_loss_decreases(self):
        obs_dim = 54  # 4 + 10*5
        policy = PolicyNetwork(obs_dim, hidden_dim=32, num_layers=1)
        trainer = BCTrainer(policy, lr=1e-3, batch_size=16)

        # Dummy data: obs -> random action labels
        np.random.seed(42)
        obs = np.random.randn(100, obs_dim).astype(np.float32)
        actions = np.random.randint(0, 9, size=100).astype(np.int64)

        losses, _normalizer = trainer.train(obs, actions, num_epochs=20, verbose=False)
        assert losses[-1] < losses[0], "Loss should decrease"

    def test_overfits_small_data(self):
        """Should overfit to small dataset."""
        obs_dim = 54  # 4 + 10*5
        policy = PolicyNetwork(obs_dim, hidden_dim=64, num_layers=2)
        trainer = BCTrainer(policy, lr=1e-3, batch_size=8)

        # Very small data — all same action
        obs = np.random.randn(10, obs_dim).astype(np.float32)
        actions = np.full(10, 4, dtype=np.int64)

        losses, _normalizer = trainer.train(obs, actions, num_epochs=100, verbose=False)
        assert losses[-1] < 0.1, f"Should overfit, got loss={losses[-1]}"


class TestWorldModelTrainer:
    def test_loss_decreases(self):
        obs_dim = 54  # 4 + 10*5
        wm = AttentionWorldModel(embed_dim=32, num_heads=4, num_layers=1,
                                  max_vehicles=11, num_actions=9,
                                  ego_features=4, features_per_neighbor=5)
        trainer = WorldModelTrainer(wm, lr=1e-3, batch_size=16)

        np.random.seed(42)
        obs = np.random.randn(100, obs_dim).astype(np.float32)
        actions = np.random.randint(0, 9, size=100).astype(np.int64)
        # next_obs = obs + small noise (easy prediction task)
        next_obs = obs + np.random.randn(100, obs_dim).astype(np.float32) * 0.1

        losses = trainer.train(obs, actions, next_obs,
                               num_epochs=20, verbose=False)
        assert losses[-1] < losses[0], "WM loss should decrease"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
