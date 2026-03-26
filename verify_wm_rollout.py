#!/usr/bin/env python3
"""Compare hybrid WM rollout (exact ego + WM NPC) against true simulator.

Shows gap accuracy which is what matters for collision avoidance.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import torch
from utils.config import Config, VehicleConfig
from utils.types import Action, LongitudinalAction
from environment.simulator import Simulator
from agents.expert import ExpertAgent
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
    vc = cfg.vehicle

    expert = ExpertAgent(obs_config=cfg.obs, num_lanes=1)
    sim_c = Simulator(cfg, seed=42)
    collector = DataCollector(sim_c, expert)
    trajs = collector.collect_episodes(30)
    obs_data, act_data, _, nobs_data, _ = trajectories_to_arrays(trajs)

    wm = AttentionWorldModel(embed_dim=64, num_heads=4, num_layers=2,
                              max_vehicles=5, num_actions=9)
    wm_trainer = WorldModelTrainer(wm, lr=3e-4, batch_size=64)
    wm_trainer.train(obs_data, act_data, nobs_data, num_epochs=80, verbose=False)
    normalizer = wm_trainer.normalizer
    print("WM trained.\n")

    # Find tight-gap seed
    print("Scanning for tight-gap scenario...")
    best_seed, best_gap = 99, 999
    for s in range(200):
        sim_s = Simulator(cfg, seed=s)
        sim_s.reset()
        ahead = [n for n in sim_s.npc_states if n.x > sim_s.ego.x]
        if ahead:
            g = min(n.x - sim_s.ego.x for n in ahead)
            if 25 < g < 40 and g < best_gap:
                best_gap, best_seed = g, s
    print(f"  Using seed={best_seed} with initial gap={best_gap:.1f}m\n")

    sim = Simulator(cfg, seed=best_seed)
    obs_true = sim.reset()
    k = (obs_dim - 3) // 4

    print(f"Initial: ego x={sim.ego.x:.1f} speed={sim.ego.speed:.1f}")
    for n in sim.npc_states:
        print(f"  NPC {n.vehicle_id}: x={n.x:.1f} speed={n.speed:.1f}")
    print()

    # Init hybrid state: exact ego + absolute NPC positions
    ego_speed = obs_true[0]
    ego_x = obs_true[2]
    npc_abs_x = np.zeros(k)
    npc_abs_speed = np.zeros(k)
    npc_exists = np.zeros(k)
    for ni in range(k):
        base = 3 + ni * 4
        npc_abs_x[ni] = ego_x + obs_true[base]
        npc_abs_speed[ni] = ego_speed + obs_true[base + 2]
        npc_exists[ni] = obs_true[base + 3]

    wm.eval()
    dt = cfg.sim.dt
    horizon = 40

    print(f"{'Step':>4} | {'Action':>10} | "
          f"{'True spd':>8} {'Hyb spd':>8} {'err':>6} | "
          f"{'True gap':>8} {'Hyb gap':>8} {'err':>8}")
    print("-" * 85)

    with torch.no_grad():
        for step in range(horizon):
            action = expert.act(obs_true)
            action_idx = action.to_index()

            # --- True simulator step ---
            obs_true_next, _, done, info = sim.step(action)

            # --- Hybrid step ---
            prev_speed = ego_speed
            prev_x = ego_x

            # Exact ego kinematics
            if action.longitudinal == LongitudinalAction.ACCELERATE:
                accel = vc.max_acceleration
            elif action.longitudinal == LongitudinalAction.DECELERATE:
                accel = -vc.max_deceleration
            else:
                accel = 0.0
            ego_speed = np.clip(ego_speed + accel * dt,
                                vc.min_speed, vc.max_speed)
            ego_x += ego_speed * dt

            # Build relative obs for WM
            obs_for_wm = np.zeros(obs_dim, dtype=np.float32)
            obs_for_wm[0] = prev_speed
            obs_for_wm[1] = 0  # lane
            obs_for_wm[2] = prev_x
            for ni in range(k):
                base = 3 + ni * 4
                obs_for_wm[base] = npc_abs_x[ni] - prev_x
                obs_for_wm[base + 1] = 0  # same lane
                obs_for_wm[base + 2] = npc_abs_speed[ni] - prev_speed
                obs_for_wm[base + 3] = npc_exists[ni]

            # WM prediction
            obs_norm = normalizer.transform(obs_for_wm.reshape(1, -1))
            obs_t = torch.FloatTensor(obs_norm)
            act_t = torch.LongTensor([action_idx])
            pred_norm = wm(obs_t, act_t)
            pred_real = normalizer.inverse_transform(pred_norm.cpu().numpy())[0]

            # Extract NPC deltas
            ego_delta_x = ego_x - prev_x
            ego_delta_speed = ego_speed - prev_speed
            for ni in range(k):
                base = 3 + ni * 4
                if npc_exists[ni] < 0.5:
                    continue
                pred_rel_x = pred_real[base]
                pred_rel_speed = pred_real[base + 2]
                cur_rel_x = obs_for_wm[base]
                cur_rel_speed = obs_for_wm[base + 2]

                delta_rel_x = pred_rel_x - cur_rel_x
                delta_rel_speed = pred_rel_speed - cur_rel_speed

                npc_abs_x[ni] += delta_rel_x + ego_delta_x
                npc_abs_speed[ni] += delta_rel_speed + ego_delta_speed

            # Find gap to nearest ahead
            true_gap = None
            hyb_gap = None
            for ni in range(k):
                base = 3 + ni * 4
                if obs_true_next[base + 3] > 0.5 and obs_true_next[base] > 0:
                    if true_gap is None or obs_true_next[base] < true_gap:
                        true_gap = obs_true_next[base]
                        hyb_gap = npc_abs_x[ni] - ego_x

            gap_str = ""
            if true_gap is not None and hyb_gap is not None:
                gap_str = (f"{true_gap:>8.1f} {hyb_gap:>8.1f} "
                           f"{abs(hyb_gap - true_gap):>8.1f}")
            else:
                gap_str = f"{'N/A':>8} {'N/A':>8} {'N/A':>8}"

            print(f"{step+1:>4} | {action.longitudinal.name:>10} | "
                  f"{obs_true_next[0]:>8.1f} {ego_speed:>8.1f} "
                  f"{abs(ego_speed - obs_true_next[0]):>6.2f} | {gap_str}")

            obs_true = obs_true_next
            if done:
                print(f"\n  True sim ended: collision={info.get('collision')}")
                break

    print(f"\nFinal errors at step {step+1}:")
    print(f"  Speed: {abs(ego_speed - obs_true[0]):.2f} m/s")
    print(f"  Position: {abs(ego_x - obs_true[2]):.1f} m")
    if true_gap is not None and hyb_gap is not None:
        print(f"  Gap: {abs(hyb_gap - true_gap):.1f} m")


if __name__ == "__main__":
    main()
