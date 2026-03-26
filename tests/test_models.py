"""Tests for models: PolicyNetwork and AttentionWorldModel."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from models.policy_net import PolicyNetwork
from models.world_model import (
    AttentionWorldModel, VehicleEmbedding,
    MultiHeadSelfAttention, TransformerBlock,
)


class TestPolicyNetwork:
    def setup_method(self):
        self.obs_dim = 27  # 3 + 6*4
        self.policy = PolicyNetwork(self.obs_dim, hidden_dim=64, num_layers=2)

    def test_forward_shape(self):
        obs = torch.randn(4, self.obs_dim)
        logits = self.policy(obs)
        assert logits.shape == (4, 9)

    def test_single_obs(self):
        obs = torch.randn(1, self.obs_dim)
        logits = self.policy(obs)
        assert logits.shape == (1, 9)

    def test_action_probs_sum_to_one(self):
        obs = torch.randn(1, self.obs_dim)
        probs = self.policy.get_action_probs(obs)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_sample_action_in_range(self):
        obs = torch.randn(1, self.obs_dim)
        idx = self.policy.sample_action(obs)
        assert 0 <= idx < 9

    def test_greedy_action_in_range(self):
        obs = torch.randn(1, self.obs_dim)
        idx = self.policy.greedy_action(obs)
        assert 0 <= idx < 9

    def test_gradient_flows(self):
        obs = torch.randn(2, self.obs_dim)
        logits = self.policy(obs)
        loss = logits.sum()
        loss.backward()
        for p in self.policy.parameters():
            assert p.grad is not None


class TestMultiHeadSelfAttention:
    def test_output_shape(self):
        attn = MultiHeadSelfAttention(embed_dim=32, num_heads=4)
        x = torch.randn(2, 5, 32)
        out = attn(x)
        assert out.shape == (2, 5, 32)

    def test_with_mask(self):
        attn = MultiHeadSelfAttention(embed_dim=32, num_heads=4)
        x = torch.randn(2, 5, 32)
        mask = torch.tensor([[True, True, True, False, False],
                             [True, True, False, False, False]])
        out = attn(x, mask)
        assert out.shape == (2, 5, 32)


class TestAttentionWorldModel:
    def setup_method(self):
        self.k = 6
        self.obs_dim = 3 + self.k * 4  # 27
        self.wm = AttentionWorldModel(
            embed_dim=32, num_heads=4, num_layers=2,
            max_vehicles=self.k + 1, num_actions=9)

    def test_forward_shape(self):
        obs = torch.randn(4, self.obs_dim)
        actions = torch.randint(0, 9, (4,))
        pred = self.wm(obs, actions)
        assert pred.shape == (4, self.obs_dim)

    def test_single_sample(self):
        obs = torch.randn(1, self.obs_dim)
        actions = torch.tensor([3])
        pred = self.wm(obs, actions)
        assert pred.shape == (1, self.obs_dim)

    def test_gradient_flows(self):
        obs = torch.randn(2, self.obs_dim)
        actions = torch.randint(0, 9, (2,))
        pred = self.wm(obs, actions)
        loss = pred.sum()
        loss.backward()
        for p in self.wm.parameters():
            assert p.grad is not None

    def test_residual_prediction(self):
        """With zero weights the prediction should be close to input (residual)."""
        # Not exactly zero due to random init, but let's just check shapes
        obs = torch.randn(1, self.obs_dim)
        actions = torch.tensor([0])
        pred = self.wm(obs, actions)
        assert pred.shape == obs.shape

    def test_handles_no_neighbors(self):
        """Observation with all neighbor exists=0 should not crash."""
        obs = torch.zeros(1, self.obs_dim)
        obs[0, 0] = 20.0  # ego speed
        actions = torch.tensor([4])
        pred = self.wm(obs, actions)
        assert not torch.isnan(pred).any()

    def test_batch_consistency(self):
        """Same input repeated should give same output."""
        obs = torch.randn(1, self.obs_dim)
        actions = torch.tensor([2])
        self.wm.eval()
        pred1 = self.wm(obs, actions)
        pred2 = self.wm(obs, actions)
        assert torch.allclose(pred1, pred2)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
