#!/usr/bin/env python3
"""
Simplified single-lane comparison for debugging.

Setup:
  - 1 lane, no lane changes allowed
  - 2 NPCs: one ahead (constant speed), one behind (constant speed)
  - Ego controls only longitudinal: brake / keep / accelerate
  - All agents should achieve 0% collision rate

This isolates the longitudinal control problem and removes all
stochasticity from the environment.
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


def make_simple_config():
    """Single lane, 2 IDM NPCs."""
    cfg = Config()
    cfg.road.num_lanes = 1
    cfg.traffic.num_npcs = 2
    cfg.traffic.spawn_range = 80.0
    cfg.traffic.despawn_distance = 500.0
    cfg.traffic.min_spawn_gap = 20.0
    # All NPCs use IDM (choice 0 and 1 both map to IDM now)
    cfg.traffic.behavior_mix = [0.0, 1.0, 0.0]
    cfg.traffic.personality_std = 0.0
    cfg.sim.episode_steps = 200
    cfg.sim.collision_reward = -50.0
    cfg.sim.speed_reward_scale = 1.0
    cfg.sim.lane_change_penalty = 0.0
    cfg.sim.hard_brake_penalty = -0.5
    cfg.obs.k_neighbors = 4
    return cfg



def main():
    parser = argparse.ArgumentParser(description="Simplified 1-lane debug comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collect-episodes", type=int, default=30)
    parser.add_argument("--bc-epochs", type=int, default=50)
    parser.add_argument("--wm-epochs", type=int, default=80)
    parser.add_argument("--rl-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--planner-horizon", type=int, default=30)
    parser.add_argument("--planner-rollouts", type=int, default=50)
    args = parser.parse_args()

    cfg = make_simple_config()
    obs_dim = 3 + cfg.obs.k_neighbors * 4  # 19

    mc = ModelConfig()
    mc.planner_horizon = args.planner_horizon
    mc.planner_num_rollouts = args.planner_rollouts

    print("=" * 60)
    print("SIMPLIFIED DEBUG: 1 lane, constant-speed NPCs")
    print(f"  Lanes: {cfg.road.num_lanes}  NPCs: {cfg.traffic.num_npcs}  "
          f"Behavior: constant speed only")
    print(f"  obs_dim: {obs_dim}")
    print("  Expected: ALL agents 0% collision rate")
    print("=" * 60)

    # ---- 1. Expert ----
    print("\n[1/5] Expert Agent")
    expert = ExpertAgent(
        obs_config=cfg.obs,
        num_lanes=cfg.road.num_lanes,
        safe_distance=20.0,
        desired_speed=28.0,
    )

    # ---- 2. Collect expert data ----
    print(f"\n[2/5] Collecting {args.collect_episodes} expert episodes...")
    sim_collect = Simulator(cfg, seed=args.seed)
    collector = DataCollector(sim_collect, expert)
    trajectories = collector.collect_episodes(args.collect_episodes)
    obs_data, act_data, rew_data, nobs_data, done_data = \
        trajectories_to_arrays(trajectories)
    expert_rewards = [t.total_reward for t in trajectories]
    print(f"  {len(obs_data)} transitions, avg reward: {np.mean(expert_rewards):.1f}")

    # Analyze action distribution
    unique, counts = np.unique(act_data, return_counts=True)
    print("  Action distribution:")
    from utils.types import Action
    for u, c in zip(unique, counts):
        a = Action.from_index(u)
        pct = 100.0 * c / len(act_data)
        print(f"    {a.longitudinal.name:>12} / {a.lateral.name:<6}  "
              f"({u}): {c:>5} ({pct:.1f}%)")

    # ---- 3. Train BC ----
    print(f"\n[3/5] Training Behavior Cloning ({args.bc_epochs} epochs)...")
    bc_policy = PolicyNetwork(obs_dim, hidden_dim=64, num_layers=2)
    bc_trainer = BCTrainer(bc_policy, lr=3e-4, batch_size=64)
    bc_losses = bc_trainer.train(obs_data, act_data,
                                  num_epochs=args.bc_epochs, verbose=True)
    bc_agent = BCAgent(bc_policy, deterministic=True)

    # Check what BC predicts
    print("  BC action predictions on training data sample:")
    bc_policy.eval()
    with torch.no_grad():
        sample_obs = torch.FloatTensor(obs_data[:100])
        preds = bc_policy(sample_obs).argmax(dim=-1).numpy()
    unique_p, counts_p = np.unique(preds, return_counts=True)
    for u, c in zip(unique_p, counts_p):
        a = Action.from_index(u)
        print(f"    {a.longitudinal.name:>12} / {a.lateral.name:<6}  "
              f"({u}): {c:>3}")

    # ---- 4. Train World Model ----
    print(f"\n[4/5] Training World Model ({args.wm_epochs} epochs)...")
    world_model = AttentionWorldModel(
        embed_dim=mc.wm_embed_dim, num_heads=mc.wm_num_heads,
        num_layers=mc.wm_num_layers,
        max_vehicles=cfg.obs.k_neighbors + 1, num_actions=9)
    wm_trainer = WorldModelTrainer(world_model, lr=3e-4, batch_size=64)
    wm_losses = wm_trainer.train(obs_data, act_data, nobs_data,
                                  num_epochs=args.wm_epochs, verbose=True)

    # Validate WM: single-step prediction error in real space
    print("  WM single-step prediction test (first 100 samples):")
    world_model.eval()
    with torch.no_grad():
        test_obs = wm_trainer.normalizer.transform(obs_data[:100])
        test_nobs = wm_trainer.normalizer.transform(nobs_data[:100])
        test_obs_t = torch.FloatTensor(test_obs)
        test_act_t = torch.LongTensor(act_data[:100])
        pred_norm = world_model(test_obs_t, test_act_t).numpy()
        # Denormalize
        pred_real = wm_trainer.normalizer.inverse_transform(pred_norm)
        actual_real = nobs_data[:100]
        errors = np.abs(pred_real - actual_real)
        print(f"    ego_speed MAE: {errors[:, 0].mean():.3f} m/s")
        print(f"    ego_lane  MAE: {errors[:, 1].mean():.3f}")
        print(f"    ego_x     MAE: {errors[:, 2].mean():.3f} m")
        if obs_dim > 3:
            print(f"    neighbor  MAE: {errors[:, 3:].mean():.3f}")

    # Validate WM: multi-step rollout vs real simulator
    print("  WM multi-step rollout test (30 steps, KEEP action):")
    sim_val = Simulator(cfg, seed=args.seed + 50)
    obs_val = sim_val.reset()
    obs_wm = torch.FloatTensor(
        wm_trainer.normalizer.transform(obs_val.reshape(1, -1))).to(
        next(world_model.parameters()).device)
    keep_action = torch.LongTensor([4])  # KEEP/KEEP = index 4
    world_model.eval()
    with torch.no_grad():
        for step in range(30):
            # Real sim step
            real_action = Action.from_index(4)
            obs_val_next, _, done_val, _ = sim_val.step(real_action)
            # WM step
            obs_wm = world_model(obs_wm, keep_action)
            # Compare in real space
            pred_real_step = wm_trainer.normalizer.inverse_transform(
                obs_wm.cpu().numpy())
            if step in [0, 4, 9, 14, 19, 29]:
                err_speed = abs(pred_real_step[0, 0] - obs_val_next[0])
                err_x = abs(pred_real_step[0, 2] - obs_val_next[2])
                # Neighbor rel_x error (first neighbor)
                err_neigh = abs(pred_real_step[0, 3] - obs_val_next[3]) if obs_dim > 3 else 0
                print(f"    step {step+1:2d}: speed_err={err_speed:.2f} "
                      f"x_err={err_x:.1f} neigh_rel_x_err={err_neigh:.1f}")
            obs_val = obs_val_next
            if done_val:
                print(f"    (real sim ended at step {step+1})")
                break

    # ---- 5. Train RL (warm-started from BC policy) ----
    print(f"\n[5/5] Training RL ({args.rl_episodes} episodes, warm-start from BC)...")
    # Copy BC policy weights as starting point
    import copy
    rl_policy = copy.deepcopy(bc_policy)
    sim_rl = Simulator(cfg, seed=args.seed + 1)
    # Lower lr to not destroy BC knowledge, lower entropy since policy is already good
    rl_trainer = RLTrainer(rl_policy, sim_rl, lr=1e-4, gamma=0.99,
                            entropy_coef=0.005)
    rl_rewards = rl_trainer.train(num_episodes=args.rl_episodes, verbose=True)
    rl_agent = BCAgent(rl_policy, deterministic=True)

    # ---- Build planners ----
    print("\nBuilding planners...")
    sim_planner = Simulator(cfg, seed=args.seed + 2)
    planner_true = PlannerTrueModel(sim_planner, mc, seed=args.seed)
    planner_learned = PlannerLearnedModel(
        world_model, mc, cfg.sim, cfg.vehicle,
        road_config=cfg.road,
        normalizer=wm_trainer.normalizer, seed=args.seed)

    # ---- Evaluate ----
    print(f"\n{'='*60}")
    print(f"Evaluating all agents ({args.eval_episodes} episodes each)...")
    print(f"{'='*60}")

    agents = {
        "Expert": expert,
        "Behavior Cloning": bc_agent,
        "Planner (true)": planner_true,
        "Planner (learned WM)": planner_learned,
        "RL (REINFORCE)": rl_agent,
    }

    results = {}
    for name, agent in agents.items():
        if name == "Planner (true)":
            evaluator = Evaluator(sim_planner, num_episodes=args.eval_episodes)
        else:
            eval_sim = Simulator(cfg, seed=args.seed + 100)
            evaluator = Evaluator(eval_sim, num_episodes=args.eval_episodes)
        results[name] = evaluator.evaluate(agent)
        m = results[name]
        status = "PASS" if m.collision_rate == 0.0 else "FAIL"
        print(f"  [{status}] {name}: collision={m.collision_rate:.0%}  "
              f"reward={m.avg_reward:.1f}  speed={m.avg_speed:.1f}")

    print(f"\n{'='*60}")
    print("FULL RESULTS")
    print(f"{'='*60}")
    Evaluator.print_comparison(results)

    # Final verdict
    print(f"\n{'='*60}")
    all_pass = all(m.collision_rate == 0.0 for m in results.values())
    if all_pass:
        print("ALL AGENTS PASS — 0% collision rate")
        print("Ready to add complexity back (lane changes, IDM/MOBIL traffic)")
    else:
        failed = [n for n, m in results.items() if m.collision_rate > 0.0]
        print(f"FAILURES: {', '.join(failed)}")
        print("Investigate these before adding complexity.")
    print("=" * 60)


if __name__ == "__main__":
    main()
