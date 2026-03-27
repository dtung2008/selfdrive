#!/usr/bin/env python3
"""Compare hybrid WM rollout (exact ego + WM NPC) against true simulator.

The "hybrid rollout" strategy uses exact analytical kinematics for the ego
vehicle (acceleration/deceleration physics are known perfectly) while relying
on the world model only to predict NPC *delta* movements.  This isolates the
WM's NPC-prediction error from any ego-prediction error.

At each step:
  1. Ego state is updated with exact kinematic equations (no WM involved).
  2. The WM receives a relative-coordinate observation and predicts the next
     relative observation.  The *change* in each NPC's relative position/speed
     is extracted and converted back to absolute coordinates.
  3. The resulting hybrid gap (ego-to-NPC distance) is compared against the
     true simulator's gap to quantify how much the WM's NPC predictions drift.

The script deliberately seeks out a "tight-gap" seed (initial gap 25-40 m)
because small gap errors matter most when they can cause false-positive or
false-negative collision predictions during planning.
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
    obs_dim = cfg.obs.ego_features + cfg.obs.k_neighbors * cfg.obs.features_per_neighbor
    vc = cfg.vehicle

    expert = ExpertAgent(obs_config=cfg.obs, num_lanes=1)
    sim_c = Simulator(cfg, seed=42)
    collector = DataCollector(sim_c, expert)
    trajs = collector.collect_episodes(30)
    obs_data, act_data, _, nobs_data, _ = trajectories_to_arrays(trajs)

    wm = AttentionWorldModel(embed_dim=64, num_heads=4, num_layers=2,
                              max_vehicles=5, num_actions=9,
                              ego_features=cfg.obs.ego_features,
                              features_per_neighbor=cfg.obs.features_per_neighbor)
    wm_trainer = WorldModelTrainer(wm, lr=3e-4, batch_size=64)
    wm_trainer.train(obs_data, act_data, nobs_data, num_epochs=80, verbose=False)
    normalizer = wm_trainer.normalizer
    print("WM trained.\n")

    # --- Tight-gap seed scanning ---
    # We scan 200 seeds looking for one where the nearest NPC ahead starts
    # 25-40 m in front of ego.  This "tight gap" range is the danger zone:
    # close enough that even small WM prediction errors could flip a
    # collision/no-collision decision in the planner.  We pick the seed with
    # the smallest gap in that band for the most demanding test.
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

    # --- Initialize hybrid state ---
    # We maintain absolute NPC positions and speeds, reconstructed from the
    # simulator's relative observation (obs gives rel_x and rel_speed for each
    # neighbor slot).  On each step we will update these using WM deltas.
    ego_speed = obs_true[0]
    ego_x = obs_true[2]
    npc_abs_x = np.zeros(k)       # absolute longitudinal position per slot
    npc_abs_speed = np.zeros(k)    # absolute speed per slot
    npc_exists = np.zeros(k)       # 1.0 if this neighbor slot is occupied
    for ni in range(k):
        base = 3 + ni * 4
        # Convert from relative (ego-centric) to absolute coordinates
        npc_abs_x[ni] = ego_x + obs_true[base]           # rel_x -> abs_x
        npc_abs_speed[ni] = ego_speed + obs_true[base + 2]  # rel_speed -> abs
        npc_exists[ni] = obs_true[base + 3]

    wm.eval()
    dt = cfg.sim.dt
    horizon = 40  # run 40 steps to see how errors accumulate over time

    # Output table columns:
    #   Step     -- simulation timestep
    #   Action   -- expert's longitudinal action (ACC/KP/BRK)
    #   True spd -- ego speed from the real simulator
    #   Hyb spd  -- ego speed from exact kinematics (should match True perfectly)
    #   err      -- absolute speed error (should be ~0 since ego is analytical)
    #   True gap -- distance to nearest NPC ahead in the real simulator
    #   Hyb gap  -- same distance computed from WM-predicted NPC positions
    #   err      -- absolute gap error (this is the key metric; grows over time)
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

            # --- Extract NPC deltas from WM prediction ---
            # The WM predicts the *next* relative observation.  To recover each
            # NPC's absolute movement we compute:
            #   delta_rel_x    = pred_rel_x - cur_rel_x
            #   delta_abs_x    = delta_rel_x + ego_delta_x
            # The ego_delta_x correction is needed because the WM's output is
            # ego-relative: if ego moved forward 2 m and the NPC didn't move,
            # the predicted rel_x would decrease by 2 m, so we must add back
            # the ego displacement to get the NPC's true absolute movement.
            # The same logic applies to speed.
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

                # Convert relative delta to absolute delta and accumulate
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
