"""Tests for planner agents and evaluator."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from utils.config import Config, ModelConfig
from utils.types import Action
from environment.simulator import Simulator
from agents.planner_true import PlannerTrueModel
from agents.planner_learned import PlannerLearnedModel
from agents.expert import ExpertAgent
from models.world_model import AttentionWorldModel
from training.evaluator import Evaluator, EvalMetrics


class TestPlannerTrueModel:
    def setup_method(self):
        cfg = Config()
        cfg.traffic.num_npcs = 2
        cfg.sim.episode_steps = 30
        self.sim = Simulator(cfg, seed=42)
        mc = ModelConfig()
        mc.planner_horizon = 3
        mc.planner_num_rollouts = 5
        self.planner = PlannerTrueModel(self.sim, mc, seed=42)

    def test_returns_valid_action(self):
        obs = self.sim.reset()
        action = self.planner.act(obs)
        assert isinstance(action, Action)
        assert action.to_index() in range(9)

    def test_preserves_state(self):
        """Planner should restore simulator state after planning."""
        obs = self.sim.reset()
        ego_x_before = self.sim.ego.x
        _ = self.planner.act(obs)
        assert abs(self.sim.ego.x - ego_x_before) < 1e-6

    def test_runs_full_episode(self):
        """Planner should be able to run a full episode."""
        obs = self.sim.reset()
        done = False
        steps = 0
        while not done and steps < 30:
            action = self.planner.act(obs)
            obs, _, done, _ = self.sim.step(action)
            steps += 1
        assert steps > 0


class TestPlannerLearnedModel:
    def setup_method(self):
        ef, fpn, k = 4, 5, 10  # current defaults
        self.obs_dim = ef + k * fpn  # 54
        self.wm = AttentionWorldModel(
            embed_dim=32, num_heads=4, num_layers=1,
            max_vehicles=k + 1, num_actions=9,
            ego_features=ef, features_per_neighbor=fpn)
        mc = ModelConfig()
        mc.planner_horizon = 3
        mc.planner_num_rollouts = 5
        self.planner = PlannerLearnedModel(self.wm, mc, seed=42)

    def test_returns_valid_action(self):
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        action = self.planner.act(obs)
        assert isinstance(action, Action)
        assert action.to_index() in range(9)

    def test_deterministic_with_same_seed(self):
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        planner1 = PlannerLearnedModel(self.wm, seed=123)
        planner2 = PlannerLearnedModel(self.wm, seed=123)
        a1 = planner1.act(obs)
        a2 = planner2.act(obs)
        assert a1.to_index() == a2.to_index()


class TestEvaluator:
    def setup_method(self):
        cfg = Config()
        cfg.traffic.num_npcs = 2
        cfg.sim.episode_steps = 30
        self.sim = Simulator(cfg, seed=42)
        self.evaluator = Evaluator(self.sim, num_episodes=3)
        self.expert = ExpertAgent(obs_config=cfg.obs,
                                   num_lanes=cfg.road.num_lanes)

    def test_evaluate_returns_metrics(self):
        metrics = self.evaluator.evaluate(self.expert)
        assert isinstance(metrics, EvalMetrics)
        assert metrics.avg_episode_length > 0

    def test_compare(self):
        agents = {"expert": self.expert}
        results = self.evaluator.compare(agents)
        assert "expert" in results
        assert isinstance(results["expert"], EvalMetrics)

    def test_print_comparison(self):
        """Should not crash."""
        agents = {"expert": self.expert}
        results = self.evaluator.compare(agents)
        Evaluator.print_comparison(results)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
