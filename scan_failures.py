#!/usr/bin/env python3
"""Scan seeds to find episodes where an agent fails (collision or early termination).

Trains the agent once, then runs it across a range of seeds. Only outputs
debug JSON for episodes that end before max_steps (i.e., collisions or other
early terminations). Skips full-length episodes silently.

This is useful for hunting down rare failure modes without manually trying
seeds one at a time.

Usage:
    python scan_failures.py --agent planner_wm --seeds 0-100
    python scan_failures.py --agent bc --seeds 0-50 --num-lanes 1
    python scan_failures.py --agent planner_wm --seeds 0-200 --wm-epochs 80 --output-dir failures/
"""
import argparse
import json
import os
import sys
import copy
sys.path.insert(0, ".")

import numpy as np
import torch

from utils.config import Config, ModelConfig
from utils.types import Action
from environment.simulator import Simulator
from agents.expert import ExpertAgent
from agents.bc_agent import BCAgent
from agents.planner_true import PlannerTrueModel
from agents.planner_learned import PlannerLearnedModel
from models.policy_net import PolicyNetwork
from models.world_model import AttentionWorldModel
from training.data_collector import DataCollector, trajectories_to_arrays
from training.bc_trainer import BCTrainer
from training.world_model_trainer import WorldModelTrainer
from training.rl_trainer import RLTrainer


def make_config(args):
    """Build simulator config from CLI arguments."""
    cfg = Config()
    cfg.road.num_lanes = args.num_lanes
    cfg.traffic.num_npcs = args.num_npcs
    cfg.sim.episode_steps = args.max_steps
    if args.num_lanes == 1:
        cfg.traffic.behavior_mix = [0.0, 1.0, 0.0]
    else:
        cfg.traffic.behavior_mix = [0.0, 0.3, 0.7]
    return cfg


def build_agent(args, cfg, obs_dim):
    """Build the requested agent, training if necessary.

    Returns:
        (agent, expert, agent_info) tuple. The expert is always built for
        comparison during recording.
    """
    expert = ExpertAgent(obs_config=cfg.obs, num_lanes=cfg.road.num_lanes)
    info = {"agent_type": args.agent}

    if args.agent == "expert":
        return expert, expert, info

    # All other agents need expert demonstration data
    print(f"Collecting {args.collect_episodes} expert episodes for training...")
    sim_c = Simulator(cfg, seed=args.train_seed)
    collector = DataCollector(sim_c, expert)
    trajs = collector.collect_episodes(args.collect_episodes)
    obs_data, act_data, rew_data, nobs_data, done_data = trajectories_to_arrays(trajs)
    info["train_transitions"] = len(obs_data)

    if args.agent == "bc":
        print(f"Training BC ({args.bc_epochs} epochs)...")
        policy = PolicyNetwork(obs_dim, hidden_dim=128, num_layers=2)
        trainer = BCTrainer(policy, lr=3e-4, batch_size=64)
        losses, normalizer = trainer.train(obs_data, act_data,
                                           num_epochs=args.bc_epochs, verbose=True)
        info["bc_final_loss"] = losses[-1]
        return BCAgent(policy, deterministic=True, normalizer=normalizer), expert, info

    if args.agent == "rl":
        print(f"Training BC for warm-start ({args.bc_epochs} epochs)...")
        bc_policy = PolicyNetwork(obs_dim, hidden_dim=128, num_layers=2)
        bc_trainer = BCTrainer(bc_policy, lr=3e-4, batch_size=64)
        bc_trainer.train(obs_data, act_data, num_epochs=args.bc_epochs, verbose=False)
        bc_normalizer = bc_trainer.normalizer

        print(f"Training RL ({args.rl_episodes} episodes, warm-start)...")
        rl_policy = copy.deepcopy(bc_policy)
        sim_rl = Simulator(cfg, seed=args.train_seed + 1)
        rl_trainer = RLTrainer(rl_policy, sim_rl, lr=1e-4, gamma=0.99, entropy_coef=0.005)
        rl_trainer.obs_normalizer = bc_normalizer
        rewards = rl_trainer.train(num_episodes=args.rl_episodes, verbose=True)
        info["rl_final_avg_reward"] = float(np.mean(rewards[-50:]))
        return BCAgent(rl_policy, deterministic=True, normalizer=bc_normalizer), expert, info

    if args.agent == "planner_true":
        mc = ModelConfig()
        mc.planner_horizon = args.planner_horizon
        mc.planner_num_rollouts = args.planner_rollouts
        # planner_true needs a sim per seed, so we return a factory instead
        # For now, return None and build per-seed in the scan loop
        info["planner_horizon"] = mc.planner_horizon
        info["planner_rollouts"] = mc.planner_num_rollouts
        return "planner_true_factory", expert, info

    if args.agent == "planner_wm":
        print(f"Training World Model ({args.wm_epochs} epochs)...")
        mc = ModelConfig()
        mc.planner_horizon = args.planner_horizon
        mc.planner_wm_horizon = args.planner_wm_horizon
        mc.planner_num_rollouts = args.planner_rollouts
        wm = AttentionWorldModel(embed_dim=mc.wm_embed_dim, num_heads=mc.wm_num_heads,
                                  num_layers=mc.wm_num_layers,
                                  max_vehicles=mc.wm_max_vehicles, num_actions=9,
                                  ego_features=cfg.obs.ego_features,
                                  features_per_neighbor=cfg.obs.features_per_neighbor)
        wm_trainer = WorldModelTrainer(wm, lr=3e-4, batch_size=64)
        wm_losses = wm_trainer.train(obs_data, act_data, nobs_data,
                                      num_epochs=args.wm_epochs, verbose=True)
        info["wm_final_loss"] = wm_losses[-1]
        info["planner_wm_horizon"] = mc.planner_wm_horizon
        planner = PlannerLearnedModel(wm, mc, cfg.sim, cfg.vehicle,
                                       road_config=cfg.road,
                                       normalizer=wm_trainer.normalizer,
                                       obs_config=cfg.obs,
                                       seed=args.train_seed)
        return planner, expert, info

    raise ValueError(f"Unknown agent: {args.agent}")


def record_episode(sim, agent, expert, max_steps):
    """Run one episode, recording per-step debug data.

    Returns:
        (frames, info, total_steps) where total_steps is the number of
        simulation steps completed.
    """
    obs = sim.reset()
    agent.reset()
    expert.reset()
    frames = []
    done = False
    step = 0
    cumulative_reward = 0.0

    # Initial frame
    ego, npcs = sim.get_all_vehicle_states()
    frames.append({
        "step": 0,
        "ego": {"x": round(ego.x, 2), "lane": ego.lane, "speed": round(ego.speed, 2)},
        "npcs": [{"id": n.vehicle_id, "x": round(n.x, 2), "lane": n.lane,
                  "speed": round(n.speed, 2)} for n in npcs],
        "reward": 0.0,
        "cumulative_reward": 0.0,
        "collision": False,
        "action": None,
        "expert_action": None,
    })

    while not done and step < max_steps:
        action = agent.act(obs)
        expert_action = expert.act(obs)
        obs, reward, done, info = sim.step(action)
        step += 1
        cumulative_reward += reward

        ego, npcs = sim.get_all_vehicle_states()
        num_lanes = sim.road.num_lanes

        lane_gaps = {}
        for lane in range(num_lanes):
            ahead_in_lane = [n for n in npcs if n.lane == lane and n.x > ego.x]
            behind_in_lane = [n for n in npcs if n.lane == lane and n.x < ego.x]
            lane_gaps[lane] = {
                "ahead": round(min(n.x - ego.x for n in ahead_in_lane), 1) if ahead_in_lane else None,
                "behind": round(min(ego.x - n.x for n in behind_in_lane), 1) if behind_in_lane else None,
            }

        gap_ahead = lane_gaps[ego.lane]["ahead"]
        gap_behind = lane_gaps[ego.lane]["behind"]

        frames.append({
            "step": step,
            "ego": {"x": round(ego.x, 2), "lane": ego.lane, "speed": round(ego.speed, 2)},
            "npcs": [{"id": n.vehicle_id, "x": round(n.x, 2), "lane": n.lane,
                      "speed": round(n.speed, 2)} for n in npcs],
            "reward": round(reward, 3),
            "cumulative_reward": round(cumulative_reward, 1),
            "collision": info.get("collision", False),
            "action": {"lon": int(action.longitudinal), "lat": int(action.lateral)},
            "expert_action": {"lon": int(expert_action.longitudinal),
                              "lat": int(expert_action.lateral)},
            "gap_ahead": gap_ahead,
            "gap_behind": gap_behind,
            "lane_gaps": lane_gaps,
            "disagree": action.to_index() != expert_action.to_index(),
        })

    return frames, info, step


def parse_seed_range(seed_str):
    """Parse a seed range string like '0-100' or '5,10,15' or '42'.

    Returns:
        List of integer seeds.
    """
    seeds = []
    for part in seed_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    parser = argparse.ArgumentParser(
        description="Scan seeds to find and record failure episodes")
    parser.add_argument("--agent", type=str, default="planner_wm",
                        choices=["expert", "bc", "planner_true", "planner_wm", "rl"])
    parser.add_argument("--seeds", "--seed", type=str, default="0-19",
                        help="Seed range to scan (e.g., '0-100', '5,10,15', '42')")
    parser.add_argument("--train-seed", type=int, default=42,
                        help="Seed for training data collection and model training")
    parser.add_argument("--num-lanes", type=int, default=2)
    parser.add_argument("--num-npcs", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--collect-episodes", type=int, default=50)
    parser.add_argument("--bc-epochs", type=int, default=200)
    parser.add_argument("--wm-epochs", type=int, default=20)
    parser.add_argument("--rl-episodes", type=int, default=300)
    parser.add_argument("--planner-horizon", type=int, default=30)
    parser.add_argument("--planner-wm-horizon", type=int, default=4)
    parser.add_argument("--planner-rollouts", type=int, default=50)
    parser.add_argument("--output-dir", "--output", type=str, default=".",
                        help="Directory for failure JSON files (created if needed)")
    args = parser.parse_args()

    seeds = parse_seed_range(args.seeds)
    cfg = make_config(args)
    obs_dim = cfg.obs.ego_features + cfg.obs.k_neighbors * cfg.obs.features_per_neighbor
    max_steps = cfg.sim.episode_steps

    # Seed training RNGs
    import random
    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)

    # Train the agent once
    agent, expert, agent_info = build_agent(args, cfg, obs_dim)

    # Create output directory if needed
    if args.output_dir != "." and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Scan seeds
    print(f"\nScanning {len(seeds)} seeds for {args.agent} failures "
          f"(lanes={args.num_lanes}, npcs={args.num_npcs}, max_steps={max_steps})...")
    print("-" * 60)

    failures = []
    for seed in seeds:
        # Build simulator for this seed
        if args.agent == "planner_true":
            mc = ModelConfig()
            mc.planner_horizon = args.planner_horizon
            mc.planner_num_rollouts = args.planner_rollouts
            sim = Simulator(cfg, seed=seed)
            test_agent = PlannerTrueModel(sim, mc, seed=seed)
        elif args.agent == "planner_wm":
            sim = Simulator(cfg, seed=seed)
            # Reset the planner's RNG state for each seed so results are
            # independent but the WM weights stay the same
            agent.reset()
            test_agent = agent
        else:
            sim = Simulator(cfg, seed=seed)
            test_agent = agent

        frames, info, total_steps = record_episode(sim, test_agent, expert, max_steps)

        # Only output if the episode ended early (not full length)
        if total_steps < max_steps:
            collision = info.get("collision", False)
            reward = frames[-1]["cumulative_reward"] if frames else 0
            disagrees = sum(1 for f in frames if f.get("disagree", False))
            reason = "COLLISION" if collision else "EARLY_END"

            print(f"  seed={seed:>4}: [{reason}] step={total_steps}, "
                  f"reward={reward:.1f}, disagrees={disagrees}")

            # Save debug JSON
            output = {
                "version": 2,
                "config": {
                    "agent": args.agent,
                    "seed": seed,
                    "num_lanes": cfg.road.num_lanes,
                    "num_npcs": cfg.traffic.num_npcs,
                    "lane_width": cfg.road.lane_width,
                    "max_steps": max_steps,
                    "dt": cfg.sim.dt,
                },
                "agent_info": agent_info,
                "summary": {
                    "total_steps": len(frames) - 1,
                    "total_reward": reward,
                    "collision": collision,
                    "final_speed": frames[-1]["ego"]["speed"] if frames else 0,
                    "disagree_count": disagrees,
                },
                "frames": frames,
            }

            filename = f"failure_{args.agent}_seed{seed}.json"
            filepath = os.path.join(args.output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(output, f, indent=2)

            failures.append({"seed": seed, "reason": reason, "steps": total_steps,
                             "reward": reward, "file": filepath})

    # Summary
    print("-" * 60)
    if failures:
        print(f"Found {len(failures)} failure(s) out of {len(seeds)} seeds "
              f"({100*len(failures)/len(seeds):.0f}% failure rate)")
        for f in failures:
            print(f"  {f['file']}")
    else:
        print(f"All {len(seeds)} seeds completed full episodes -- no failures found.")


if __name__ == "__main__":
    main()
