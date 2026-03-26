#!/usr/bin/env python3
"""Test WM planner at different horizons to verify error compounding diagnosis.

If the WM planner works at short horizons but fails at long ones,
it confirms the multi-step error compounding problem.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import torch
from utils.config import Config, ModelConfig
from environment.simulator import Simulator
from agents.expert import ExpertAgent
from agents.planner_learned import PlannerLearnedModel
from models.policy_net import PolicyNetwork
from models.world_model import AttentionWorldModel
from training.data_collector import DataCollector, trajectories_to_arrays
from training.world_model_trainer import WorldModelTrainer


def main():
    cfg = Config()
    cfg.road.num_lanes = 1
    cfg.traffic.num_npcs = 2
    cfg.traffic.behavior_mix = [0.0, 1.0, 0.0]
    cfg.traffic.personality_std = 0.0
    cfg.sim.episode_steps = 200
    cfg.obs.k_neighbors = 4
    obs_dim = 3 + 4 * 4

    # Collect expert data and train WM
    expert = ExpertAgent(obs_config=cfg.obs, num_lanes=1)
    sim_c = Simulator(cfg, seed=42)
    collector = DataCollector(sim_c, expert)
    trajs = collector.collect_episodes(30)
    obs_data, act_data, _, nobs_data, _ = trajectories_to_arrays(trajs)

    wm = AttentionWorldModel(embed_dim=64, num_heads=4, num_layers=2,
                              max_vehicles=5, num_actions=9)
    wm_trainer = WorldModelTrainer(wm, lr=3e-4, batch_size=64)
    wm_trainer.train(obs_data, act_data, nobs_data, num_epochs=80, verbose=False)
    print("WM trained.\n")

    # Test at different horizons
    print(f"{'Horizon':>8} {'Rollouts':>8} {'ColRate':>8} {'AvgSpd':>8} {'AvgRew':>8}")
    print("-" * 48)

    for horizon in [3, 5, 10, 15, 20, 30]:
        mc = ModelConfig()
        mc.planner_wm_horizon = horizon
        mc.planner_num_rollouts = 50

        collisions = 0
        total_reward = 0.0
        total_speed = 0.0
        num_eps = 20
        num_steps = 0

        for seed in range(num_eps):
            planner = PlannerLearnedModel(wm, mc, cfg.sim,
                                           road_config=cfg.road,
                                           normalizer=wm_trainer.normalizer,
                                           seed=seed)
            sim = Simulator(cfg, seed=seed + 100)
            obs = sim.reset()
            done = False
            while not done:
                action = planner.act(obs)
                obs, r, done, info = sim.step(action)
                total_reward += r
                total_speed += sim.ego.speed
                num_steps += 1
            if info.get("collision"):
                collisions += 1

        print(f"{horizon:>8} {50:>8} {collisions/num_eps:>8.0%} "
              f"{total_speed/num_steps:>8.1f} {total_reward/num_eps:>8.1f}")


if __name__ == "__main__":
    main()
