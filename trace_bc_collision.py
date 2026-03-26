#!/usr/bin/env python3
"""Trace a BC collision episode step-by-step, comparing BC vs Expert decisions."""
import sys
sys.path.insert(0, ".")
import numpy as np
import torch
from utils.config import Config
from environment.simulator import Simulator
from agents.expert import ExpertAgent
from agents.bc_agent import BCAgent
from models.policy_net import PolicyNetwork
from training.data_collector import DataCollector, trajectories_to_arrays
from training.bc_trainer import BCTrainer


def main():
    cfg = Config()
    cfg.road.num_lanes = 1
    cfg.traffic.num_npcs = 2
    cfg.traffic.behavior_mix = [1.0, 0.0, 0.0]
    cfg.traffic.personality_std = 0.0
    cfg.sim.episode_steps = 200
    cfg.obs.k_neighbors = 4
    obs_dim = 3 + 4 * 4

    # Train BC
    expert = ExpertAgent(obs_config=cfg.obs, num_lanes=1)
    sim_c = Simulator(cfg, seed=42)
    collector = DataCollector(sim_c, expert)
    trajs = collector.collect_episodes(30)
    obs_data, act_data, _, _, _ = trajectories_to_arrays(trajs)

    bc_policy = PolicyNetwork(obs_dim, hidden_dim=64, num_layers=2)
    trainer = BCTrainer(bc_policy, lr=3e-4, batch_size=64)
    trainer.train(obs_data, act_data, num_epochs=50, verbose=False)
    bc_agent = BCAgent(bc_policy, deterministic=True)

    # Find a crashing episode
    for seed in range(200):
        sim = Simulator(cfg, seed=seed)
        obs = sim.reset()
        done = False
        while not done:
            action = bc_agent.act(obs)
            obs, r, done, info = sim.step(action)
        if info.get("collision"):
            print(f"=== Found collision at seed={seed}, replaying ===\n")

            # Replay with full trace
            sim2 = Simulator(cfg, seed=seed)
            obs2 = sim2.reset()
            print(f"Initial: ego x={sim2.ego.x:.1f} speed={sim2.ego.speed:.1f}")
            for n in sim2.npc_states:
                print(f"  NPC {n.vehicle_id}: x={n.x:.1f} speed={n.speed:.1f}")
            print()

            done2 = False
            step = 0
            while not done2:
                action_bc = bc_agent.act(obs2)
                action_ex = expert.act(obs2)

                obs2, r, done2, info2 = sim2.step(action_bc)
                step += 1

                ahead = [n for n in sim2.npc_states
                         if n.x > sim2.ego.x and n.lane == sim2.ego.lane]
                behind = [n for n in sim2.npc_states
                          if n.x <= sim2.ego.x and n.lane == sim2.ego.lane]
                gap_ahead = min((n.x - sim2.ego.x) for n in ahead) if ahead else 999
                gap_behind = min((sim2.ego.x - n.x) for n in behind) if behind else 999

                disagree = action_bc.longitudinal != action_ex.longitudinal
                show = step <= 5 or step % 10 == 0 or gap_ahead < 15 or disagree or done2

                if show:
                    tag = "  <-- DISAGREE" if disagree else ""
                    print(f"Step {step:3d}: ego_spd={sim2.ego.speed:.1f} "
                          f"gap_ahead={gap_ahead:.1f} gap_behind={gap_behind:.1f} "
                          f"BC={action_bc.longitudinal.name:>10} "
                          f"Expert={action_ex.longitudinal.name:>10}{tag}")

                if done2:
                    print(f"\nCOLLISION at step {step}")
                    print(f"  ego: x={sim2.ego.x:.1f} speed={sim2.ego.speed:.1f}")
                    for n in sim2.npc_states:
                        gap = n.x - sim2.ego.x
                        hit = (abs(gap) < (n.length / 2 + sim2.ego.length / 2)
                               and n.lane == sim2.ego.lane)
                        print(f"  NPC {n.vehicle_id}: x={n.x:.1f} "
                              f"speed={n.speed:.1f} gap={gap:.1f}"
                              f"{'  <-- HIT' if hit else ''}")
            break
    else:
        print("No BC collision found in 200 seeds")


if __name__ == "__main__":
    main()
