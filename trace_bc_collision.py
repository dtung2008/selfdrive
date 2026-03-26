#!/usr/bin/env python3
"""Trace a BC collision episode step-by-step, comparing BC vs Expert decisions.

This script performs three phases:
  1. Train a Behavioral Cloning (BC) agent on expert demonstrations collected
     in a simplified highway scenario.
  2. Scan across many random seeds to find an episode where the BC agent causes
     a collision (the expert would not).
  3. Replay that collision episode step-by-step, printing both the BC and Expert
     actions at each timestep so we can diagnose *why* the BC policy diverged
     from the expert and what led to the crash.
"""
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
    # ------------------------------------------------------------------
    # Config: simplified scenario to isolate longitudinal decision-making
    # ------------------------------------------------------------------
    cfg = Config()
    cfg.road.num_lanes = 1          # Single lane -- no lane-change decisions
    cfg.traffic.num_npcs = 2        # Only two NPC vehicles on the road
    cfg.traffic.behavior_mix = [1.0, 0.0, 0.0]  # 100% constant-speed NPCs
    cfg.traffic.personality_std = 0.0             # No randomness in NPC behavior
    cfg.sim.episode_steps = 200     # Cap episode length at 200 steps
    cfg.obs.k_neighbors = 4        # Observe the 4 nearest neighbors
    # Observation dim: 3 ego features + 4 features per neighbor * 4 neighbors
    obs_dim = 3 + 4 * 4

    # ------------------------------------------------------------------
    # Phase 1: Train a BC policy on expert demonstrations
    # ------------------------------------------------------------------
    # Collect 30 expert episodes to build the training dataset.
    expert = ExpertAgent(obs_config=cfg.obs, num_lanes=1)
    sim_c = Simulator(cfg, seed=42)
    collector = DataCollector(sim_c, expert)
    trajs = collector.collect_episodes(30)
    obs_data, act_data, _, _, _ = trajectories_to_arrays(trajs)

    # Train a small 2-layer MLP policy via supervised behavioral cloning.
    bc_policy = PolicyNetwork(obs_dim, hidden_dim=64, num_layers=2)
    trainer = BCTrainer(bc_policy, lr=3e-4, batch_size=64)
    # Capture the normalizer returned by train(); the policy was trained on
    # normalized observations, so the agent must apply the same normalization.
    _losses, normalizer = trainer.train(obs_data, act_data, num_epochs=50, verbose=False)
    bc_agent = BCAgent(bc_policy, deterministic=True, normalizer=normalizer)

    # ------------------------------------------------------------------
    # Phase 2: Scan seeds 0-199 to find one where BC causes a collision
    # ------------------------------------------------------------------
    # Each seed produces a different NPC placement / speed configuration.
    # We run the BC agent and stop at the first collision we find.
    for seed in range(200):
        sim = Simulator(cfg, seed=seed)
        obs = sim.reset()
        done = False
        while not done:
            action = bc_agent.act(obs)
            obs, r, done, info = sim.step(action)
        if info.get("collision"):
            print(f"=== Found collision at seed={seed}, replaying ===\n")

            # ----------------------------------------------------------
            # Phase 3: Replay the collision episode with a full trace
            # Re-create the same episode (same seed) and step through
            # it, printing BC vs Expert actions and proximity metrics.
            # ----------------------------------------------------------
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

                # Compute distance to the nearest same-lane NPC ahead and behind.
                # gap_ahead: distance to the closest NPC in front (999 if none).
                # gap_behind: distance to the closest NPC behind (999 if none).
                ahead = [n for n in sim2.npc_states
                         if n.x > sim2.ego.x and n.lane == sim2.ego.lane]
                behind = [n for n in sim2.npc_states
                          if n.x <= sim2.ego.x and n.lane == sim2.ego.lane]
                gap_ahead = min((n.x - sim2.ego.x) for n in ahead) if ahead else 999
                gap_behind = min((sim2.ego.x - n.x) for n in behind) if behind else 999

                # Flag timesteps where BC and Expert chose different longitudinal
                # actions -- these disagreements are the root cause of collisions.
                disagree = action_bc.longitudinal != action_ex.longitudinal
                # Print selectively: first 5 steps, every 10th step, when the gap
                # is dangerously small (<15), when BC disagrees, or at termination.
                show = step <= 5 or step % 10 == 0 or gap_ahead < 15 or disagree or done2

                if show:
                    tag = "  <-- DISAGREE" if disagree else ""
                    print(f"Step {step:3d}: ego_spd={sim2.ego.speed:.1f} "
                          f"gap_ahead={gap_ahead:.1f} gap_behind={gap_behind:.1f} "
                          f"BC={action_bc.longitudinal.name:>10} "
                          f"Expert={action_ex.longitudinal.name:>10}{tag}")

                if done2:
                    # ----- Collision report -----
                    # Print final positions and mark which NPC was hit.
                    # "gap" is the signed distance (NPC.x - ego.x); a vehicle
                    # pair overlaps when the absolute gap is less than the sum
                    # of their half-lengths and they share the same lane.
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
