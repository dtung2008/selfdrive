#!/usr/bin/env python3
"""
Self-Driving Simulator: Train and compare all agent types.

Agents compared:
  1. Expert (rule-based)
  2. Behavior Cloning (supervised, imitating expert)
  3. Planner with True Model (MPC using real simulator)
  4. Planner with Learned World Model (MPC using attention model)
  5. RL Policy (REINFORCE)

Usage:
    python run_comparison.py [--num-lanes 2] [--episodes 50] [--seed 42]
"""
import argparse
import numpy as np
import torch

from utils.config import Config, ModelConfig
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
from training.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Self-Driving Agent Comparison")
    parser.add_argument("--num-lanes", type=int, default=2)
    parser.add_argument("--num-npcs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collect-episodes", type=int, default=50,
                        help="Episodes of expert driving to collect")
    parser.add_argument("--bc-epochs", type=int, default=200)
    parser.add_argument("--wm-epochs", type=int, default=20)
    parser.add_argument("--rl-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--planner-horizon", type=int, default=30)
    parser.add_argument("--planner-rollouts", type=int, default=50)
    args = parser.parse_args()

    # ---- Config ----
    # Seed all RNGs for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = Config()
    cfg.road.num_lanes = args.num_lanes
    cfg.traffic.num_npcs = args.num_npcs
    cfg.sim.episode_steps = 300
    # Use IDM for single lane, IDM+MOBIL for multi-lane
    if args.num_lanes == 1:
        cfg.traffic.behavior_mix = [0.0, 1.0, 0.0]  # IDM only
    else:
        cfg.traffic.behavior_mix = [0.0, 0.3, 0.7]  # 30% IDM, 70% IDM+MOBIL

    mc = ModelConfig()
    mc.planner_horizon = args.planner_horizon
    mc.planner_num_rollouts = args.planner_rollouts

    obs_dim = 3 + cfg.obs.k_neighbors * 4

    print("=" * 60)
    print(f"Self-Driving Simulator Comparison")
    print(f"  Lanes: {args.num_lanes}  NPCs: {args.num_npcs}  Seed: {args.seed}")
    print("=" * 60)

    # ---- 1. Expert Agent ----
    print("\n[1/5] Expert Agent (rule-based)")
    expert = ExpertAgent(obs_config=cfg.obs, num_lanes=cfg.road.num_lanes)

    # ---- 2. Collect Expert Data ----
    print(f"\n[2/5] Collecting {args.collect_episodes} episodes of expert data...")
    sim_collect = Simulator(cfg, seed=args.seed)
    collector = DataCollector(sim_collect, expert)
    trajectories = collector.collect_episodes(args.collect_episodes)
    obs_data, act_data, rew_data, nobs_data, done_data = \
        trajectories_to_arrays(trajectories)
    print(f"  Collected {len(obs_data)} transitions "
          f"(avg reward: {np.mean([t.total_reward for t in trajectories]):.1f})")

    # ---- 3. Train Behavior Cloning ----
    print(f"\n[3/5] Training Behavior Cloning ({args.bc_epochs} epochs)...")
    bc_policy = PolicyNetwork(obs_dim, hidden_dim=128, num_layers=2)
    bc_trainer = BCTrainer(bc_policy, lr=3e-4, batch_size=64)
    bc_losses, bc_normalizer = bc_trainer.train(obs_data, act_data,
                                  num_epochs=args.bc_epochs, verbose=True)
    bc_agent = BCAgent(bc_policy, deterministic=True, normalizer=bc_normalizer)

    # ---- 4. Train World Model ----
    print(f"\n[4/5] Training Attention World Model ({args.wm_epochs} epochs)...")
    world_model = AttentionWorldModel(
        embed_dim=mc.wm_embed_dim, num_heads=mc.wm_num_heads,
        num_layers=mc.wm_num_layers,
        max_vehicles=mc.wm_max_vehicles, num_actions=9)
    wm_trainer = WorldModelTrainer(world_model, lr=3e-4, batch_size=64)
    wm_losses = wm_trainer.train(obs_data, act_data, nobs_data,
                                  num_epochs=args.wm_epochs, verbose=True)

    # ---- 5. RL Policy (warm-started from BC) ----
    print(f"\n[5/5] Training RL Policy ({args.rl_episodes} episodes, warm-start from BC)...")
    import copy
    rl_policy = copy.deepcopy(bc_policy)
    sim_rl = Simulator(cfg, seed=args.seed + 1)
    rl_trainer = RLTrainer(rl_policy, sim_rl, lr=1e-4, gamma=0.99,
                            entropy_coef=0.005)
    rl_trainer.obs_normalizer = bc_normalizer
    rl_rewards = rl_trainer.train(num_episodes=args.rl_episodes, verbose=True)
    rl_agent = BCAgent(rl_policy, deterministic=True, normalizer=bc_normalizer)

    # ---- Build Planners ----
    print("\nBuilding planners...")
    sim_planner = Simulator(cfg, seed=args.seed + 2)
    planner_true = PlannerTrueModel(sim_planner, mc, seed=args.seed)
    planner_learned = PlannerLearnedModel(world_model, mc, cfg.sim,
                                           cfg.vehicle,
                                           road_config=cfg.road,
                                           normalizer=wm_trainer.normalizer,
                                           seed=args.seed)

    # ---- Evaluate All Agents ----
    print(f"\n{'='*60}")
    print(f"Evaluating all agents ({args.eval_episodes} episodes each)...")
    print(f"{'='*60}")

    agents = {
        "Expert (rule-based)": expert,
        "Behavior Cloning": bc_agent,
        "Planner (true model)": planner_true,
        "Planner (learned WM)": planner_learned,
        "RL (REINFORCE)": rl_agent,
    }

    # Note: planner_true needs its own simulator for rollouts,
    # but evaluator uses a separate simulator for evaluation
    results = {}
    for name, agent in agents.items():
        print(f"\n  Evaluating: {name}...")
        if name == "Planner (true model)":
            # Planner needs the same sim it plans with
            evaluator = Evaluator(sim_planner, num_episodes=args.eval_episodes)
        else:
            eval_sim = Simulator(cfg, seed=args.seed + 100)
            evaluator = Evaluator(eval_sim, num_episodes=args.eval_episodes)
        results[name] = evaluator.evaluate(agent)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    Evaluator.print_comparison(results)


if __name__ == "__main__":
    main()
