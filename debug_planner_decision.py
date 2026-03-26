"""Debug script: run WM planner and dump detailed reward breakdown at stuck steps.

Runs a full episode with the WM-based planner and dumps the planner's internal
reward breakdown (per-action scores, rollout statistics) at selected timesteps.
"Stuck" steps are auto-detected when the ego vehicle's speed stays below the
desired cruising speed for 5 or more consecutive steps, indicating the planner
may be making poor decisions (e.g., braking behind a slow NPC without
attempting a lane change).  You can also manually specify step numbers via
--debug-steps to inspect arbitrary decisions.

Usage examples:
    python debug_planner_decision.py --seed 42 --debug-steps stuck
    python debug_planner_decision.py --seed 42 --debug-steps 50,100,150
"""
import argparse
import numpy as np
import torch
from environment.simulator import Simulator
from models.world_model import AttentionWorldModel
from training.data_collector import DataCollector, trajectories_to_arrays
from training.world_model_trainer import WorldModelTrainer
from agents.expert import ExpertAgent
from agents.planner_learned import PlannerLearnedModel
from utils.config import Config, ModelConfig
from utils.types import Action
from record_debug import make_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-lanes", type=int, default=2)
    parser.add_argument("--num-npcs", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--wm-epochs", type=int, default=20)
    parser.add_argument("--planner-wm-horizon", type=int, default=4)
    parser.add_argument("--collect-episodes", type=int, default=50)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--debug-steps", type=str, default="",
                        help="Comma-separated step numbers to dump, or 'stuck' for auto-detect")
    args = parser.parse_args()

    cfg = make_config(args)
    obs_dim = 3 + cfg.obs.k_neighbors * 4

    # Seed all RNGs for reproducibility
    import random
    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)

    # Collect training data
    print("Collecting expert data...")
    sim_c = Simulator(cfg, seed=args.train_seed)
    expert = ExpertAgent(obs_config=cfg.obs, num_lanes=cfg.road.num_lanes)
    collector = DataCollector(sim_c, expert)
    trajs = collector.collect_episodes(args.collect_episodes)
    obs_data, act_data, rew_data, nobs_data, done_data = trajectories_to_arrays(trajs)

    # Train WM
    mc = ModelConfig()
    mc.planner_wm_horizon = args.planner_wm_horizon
    print(f"Training WM ({args.wm_epochs} epochs)...")
    wm = AttentionWorldModel(embed_dim=mc.wm_embed_dim, num_heads=mc.wm_num_heads,
                              num_layers=mc.wm_num_layers,
                              max_vehicles=mc.wm_max_vehicles, num_actions=9)
    wm_trainer = WorldModelTrainer(wm, lr=3e-4, batch_size=64)
    wm_trainer.train(obs_data, act_data, nobs_data, num_epochs=args.wm_epochs, verbose=True)

    # Create planner
    planner = PlannerLearnedModel(wm, mc, cfg.sim, cfg.vehicle,
                                   road_config=cfg.road,
                                   normalizer=wm_trainer.normalizer,
                                   seed=args.seed)

    # Parse debug steps: three modes
    #   "stuck"       -> auto-detect stuck states at runtime (default)
    #   "10,20,30"    -> dump at exactly those step numbers
    #   "" (empty)    -> same as "stuck" (auto-detect)
    if args.debug_steps == "stuck":
        debug_steps = set()  # will be populated during the episode
        auto_detect = True
    elif args.debug_steps:
        debug_steps = set(int(x) for x in args.debug_steps.split(","))
        auto_detect = False
    else:
        debug_steps = set()
        auto_detect = True

    # Run episode
    sim = Simulator(cfg, seed=args.seed)
    obs = sim.reset()
    planner.reset()
    prev_speed = obs[0]
    prev_lane = int(round(obs[1]))
    stuck_count = 0
    total_reward = 0.0
    max_speed = 0.0
    low_speed_steps = 0
    lane_changes = 0

    for step in range(1, args.max_steps + 1):
        cur_speed = obs[0]
        cur_lane = int(round(obs[1]))
        cur_x = obs[2]

        # --- Auto-detect stuck states ---
        # A step is considered "stuck" when the ego speed is below 22 m/s
        # (roughly desired_speed - 8) AND hasn't changed from the previous
        # step (indicates the planner is holding a constant low speed).
        # We trigger a debug dump the *first* time stuck_count hits 5
        # (the moment we confirm the vehicle is stuck, not just briefly slow),
        # then again every 20 steps while still stuck to monitor if the
        # planner eventually recovers.
        if auto_detect:
            if cur_speed < 22.0 and cur_speed == prev_speed:
                stuck_count += 1
            else:
                stuck_count = 0

            if stuck_count == 5:  # just became stuck -- first dump
                debug_steps.add(step)
            if stuck_count > 0 and stuck_count % 20 == 0:  # periodic while stuck
                debug_steps.add(step)

        # Also dump 1 step before any lane change (to see why planner chose it)
        if auto_detect and cur_lane != prev_lane:
            debug_steps.add(step - 1)  # retroactive — won't help this step
            # but we can also dump current step
            debug_steps.add(step)

        should_debug = step in debug_steps

        if should_debug:
            # Setting planner.debug = True causes PlannerLearnedModel.act()
            # to print its internal reward breakdown: for each candidate action
            # it shows the mean/max reward across rollouts, helping diagnose
            # *why* the planner chose (or avoided) a particular action.
            planner.debug = True
            # Also dump the observation state
            k = (len(obs) - 3) // 4
            print(f"\n{'='*70}")
            print(f"STEP {step}: ego_speed={cur_speed:.1f} lane={cur_lane} x={cur_x:.1f}")
            print(f"NPCs visible:")
            for ni in range(k):
                base = 3 + ni * 4
                rel_x = obs[base]
                rel_lane = obs[base + 1]
                rel_speed = obs[base + 2]
                exists = obs[base + 3]
                if exists > 0.5:
                    abs_lane = cur_lane + rel_lane
                    abs_speed = cur_speed + rel_speed
                    direction = "AHEAD" if rel_x > 0 else "BEHIND"
                    print(f"  NPC{ni}: lane={abs_lane:.0f} gap={rel_x:+.1f}m ({direction}) "
                          f"speed={abs_speed:.1f} m/s")

        action = planner.act(obs)
        planner.debug = False

        obs, reward, done, info = sim.step(action)

        # Track for summary
        total_reward += reward
        max_speed = max(max_speed, cur_speed)
        if cur_speed < 22.0:
            low_speed_steps += 1
        lane_changes += (int(round(obs[1])) != cur_lane)

        prev_speed = cur_speed
        prev_lane = cur_lane

        if done:
            print(f"\nEpisode ended at step {step}: collision={info.get('collision', False)}")
            break

    # --- Summary statistics ---
    # steps           : total timesteps before episode ended (or max_steps)
    # final_speed     : ego speed at the last observed state
    # total_reward    : cumulative reward (higher = better driving)
    # max_speed       : peak speed achieved during the episode
    # lane_changes    : number of lane switches (frequent changes may signal
    #                   indecisive planning)
    # low_speed_steps : how many steps ego was below 22 m/s (stuck indicator)
    # debug_dumps     : how many times we printed a reward breakdown;
    #                   0 means the planner never got stuck
    print(f"\n{'='*70}")
    print(f"SUMMARY: steps={step}, final_speed={obs[0]:.1f}, total_reward={total_reward:.1f}")
    print(f"  max_speed={max_speed:.1f}, lane_changes={lane_changes}, low_speed_steps={low_speed_steps}/{step}")
    print(f"  debug_dumps={len(debug_steps)} (0 = never got stuck)")

if __name__ == "__main__":
    main()
