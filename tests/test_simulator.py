"""Tests for environment/simulator.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from utils.config import Config
from utils.types import Action, LongitudinalAction, LateralAction
from environment.simulator import Simulator


class TestSimulator:
    def setup_method(self):
        cfg = Config()
        cfg.traffic.num_npcs = 3  # fewer NPCs for faster tests
        cfg.sim.episode_steps = 50
        self.sim = Simulator(cfg, seed=42)

    def test_reset_returns_obs(self):
        obs = self.sim.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 1
        expected_size = 3 + self.sim.config.obs.k_neighbors * 4
        assert obs.shape[0] == expected_size

    def test_step_returns_tuple(self):
        self.sim.reset()
        action = Action()
        obs, reward, done, info = self.sim.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_advances_position(self):
        self.sim.reset()
        x_before = self.sim.ego.x
        action = Action(LongitudinalAction.KEEP, LateralAction.KEEP)
        self.sim.step(action)
        assert self.sim.ego.x > x_before

    def test_accelerate_increases_speed(self):
        self.sim.reset()
        speed_before = self.sim.ego.speed
        action = Action(LongitudinalAction.ACCELERATE, LateralAction.KEEP)
        self.sim.step(action)
        assert self.sim.ego.speed >= speed_before

    def test_episode_terminates(self):
        """Episode should end after max steps."""
        self.sim.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            action = Action()
            _, _, done, info = self.sim.step(action)
            steps += 1
        assert done
        assert steps <= self.sim.config.sim.episode_steps

    def test_collision_detection(self):
        """Manually place NPC on top of ego to trigger collision."""
        self.sim.reset()
        from utils.types import VehicleState
        # Force a collision by placing ego and NPC at same position
        self.sim.ego = VehicleState(x=100, lane=0, speed=20.0, vehicle_id=0)
        self.sim.npc_states = [
            VehicleState(x=100, lane=0, speed=20.0, vehicle_id=999)
        ]
        assert self.sim._check_collision() is True

    def test_no_collision_different_lanes(self):
        self.sim.reset()
        from utils.types import VehicleState
        self.sim.ego = VehicleState(x=100, lane=0, speed=20.0, vehicle_id=0)
        self.sim.npc_states = [
            VehicleState(x=100, lane=1, speed=20.0, vehicle_id=999)
        ]
        assert self.sim._check_collision() is False

    def test_clone_restore_state(self):
        self.sim.reset()
        snap = self.sim.clone_state()
        ego_x_before = self.sim.ego.x

        # Take some steps
        for _ in range(10):
            self.sim.step(Action(LongitudinalAction.ACCELERATE, LateralAction.KEEP))
        assert self.sim.ego.x != ego_x_before

        # Restore
        self.sim.restore_state(snap)
        assert abs(self.sim.ego.x - ego_x_before) < 1e-6

    def test_collision_gives_negative_reward(self):
        self.sim.reset()
        from utils.types import VehicleState
        # Place NPC just ahead in same lane — will collide on next step
        self.sim.ego = VehicleState(x=100, lane=0, speed=25.0, vehicle_id=0)
        self.sim.npc_states = [
            VehicleState(x=102, lane=0, speed=5.0, vehicle_id=999)
        ]
        # Force traffic manager to have this NPC
        self.sim.traffic.npcs = {
            999: (self.sim.npc_states[0],
                  self.sim.traffic._assign_behavior())
        }
        action = Action(LongitudinalAction.ACCELERATE, LateralAction.KEEP)
        _, reward, done, info = self.sim.step(action)
        if info.get("collision"):
            assert reward < 0

    def test_speed_reward_positive(self):
        """Driving at positive speed with no collision should give positive reward."""
        self.sim.reset()
        # Clear NPCs to avoid collision
        self.sim.npc_states = []
        self.sim.traffic.npcs = {}
        action = Action(LongitudinalAction.KEEP, LateralAction.KEEP)
        _, reward, _, _ = self.sim.step(action)
        assert reward > 0

    def test_cannot_step_after_done(self):
        self.sim.reset()
        self.sim._done = True
        try:
            self.sim.step(Action())
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_multi_lane_config(self):
        cfg = Config()
        cfg.road.num_lanes = 4
        cfg.traffic.num_npcs = 3
        sim = Simulator(cfg, seed=99)
        obs = sim.reset()
        assert sim.road.num_lanes == 4
        assert isinstance(obs, np.ndarray)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
