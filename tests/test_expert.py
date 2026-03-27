"""Tests for agents/expert.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from utils.config import Config
from utils.types import Action, LongitudinalAction, LateralAction
from environment.simulator import Simulator
from agents.expert import ExpertAgent


class TestExpertAgent:
    def setup_method(self):
        self.cfg = Config()
        self.cfg.traffic.num_npcs = 3
        self.cfg.sim.episode_steps = 100
        self.sim = Simulator(self.cfg, seed=42)
        self.expert = ExpertAgent(
            obs_config=self.cfg.obs,
            num_lanes=self.cfg.road.num_lanes,
        )

    def test_returns_valid_action(self):
        obs = self.sim.reset()
        action = self.expert.act(obs)
        assert isinstance(action, Action)
        assert action.to_index() in range(9)

    def test_survives_episode(self):
        """Expert should mostly complete episodes without crashing."""
        collisions = 0
        completions = 0
        for seed in range(5):
            sim = Simulator(self.cfg, seed=seed)
            obs = sim.reset()
            done = False
            while not done:
                action = self.expert.act(obs)
                obs, _, done, info = sim.step(action)
            if info.get("collision"):
                collisions += 1
            if info.get("timeout"):
                completions += 1
        # Expert should complete most episodes
        assert completions >= 2, f"Only {completions}/5 completions"

    def test_accelerates_on_empty_road(self):
        """With no neighbors, expert should accelerate or keep if near desired speed."""
        obs = np.zeros(self.sim.obs_builder.obs_size, dtype=np.float32)
        obs[0] = 15.0   # ego_speed = 15 (below desired)
        obs[1] = 0.0     # ego_lane
        obs[2] = 100.0   # ego_x
        action = self.expert.act(obs)
        assert action.longitudinal == LongitudinalAction.ACCELERATE

    def test_decelerates_near_slow_leader(self):
        """Should brake when close vehicle ahead is slower."""
        ef = self.cfg.obs.ego_features
        fpn = self.cfg.obs.features_per_neighbor
        obs = np.zeros(self.sim.obs_builder.obs_size, dtype=np.float32)
        obs[0] = 25.0   # ego speed
        obs[1] = 0.0     # ego lane
        obs[2] = 100.0   # ego x
        # First neighbor: close ahead, slower
        base = ef
        obs[base + 0] = 10.0    # rel_x = 10 (close)
        obs[base + 1] = 0.0     # same lane
        obs[base + 2] = -10.0   # rel_speed = -10 (leader is 10 m/s slower)
        obs[base + fpn - 1] = 1.0  # exists
        action = self.expert.act(obs)
        assert action.longitudinal == LongitudinalAction.DECELERATE

    def test_lane_change_to_pass(self):
        """Should change lane when blocked by very slow leader in 2-lane road."""
        ef = self.cfg.obs.ego_features
        fpn = self.cfg.obs.features_per_neighbor
        obs = np.zeros(self.sim.obs_builder.obs_size, dtype=np.float32)
        obs[0] = 25.0
        obs[1] = 0.0
        obs[2] = 100.0
        # Slow leader close ahead in lane 0
        base = ef
        obs[base + 0] = 15.0     # rel_x = 15
        obs[base + 1] = 0.0      # same lane
        obs[base + 2] = -15.0    # leader is very slow
        obs[base + fpn - 1] = 1.0  # exists
        action = self.expert.act(obs)
        # Should try to change to lane 1
        assert action.lateral == LateralAction.RIGHT


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
