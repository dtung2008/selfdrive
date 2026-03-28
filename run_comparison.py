#!/usr/bin/env python3
"""
Self-Driving Simulator: Train and compare all agent types.

Pipeline overview (5 stages):
  1. Expert agent        -- A hand-crafted, rule-based driver used as a
                            performance ceiling and as the teacher for
                            imitation learning.
  2. Data collection     -- Roll out the expert in the simulator to gather
                            (obs, action, reward, next_obs, done) transitions
                            that serve as training data for stages 3 and 4.
  3. BC training         -- Supervised behavior-cloning: a policy network
                            learns to map observations to expert actions.
  4. World-model training-- An attention-based world model learns the
                            environment dynamics (obs, action) -> next_obs so
                            that a planner can do lookahead without querying
                            the real simulator.
  5. Evaluation          -- Every agent (expert, BC, planner-true,
                            planner-learned) is evaluated on the same set
                            of episodes and the results are compared.

Agents compared:
  1. Expert (rule-based)
  2. Behavior Cloning (supervised, imitating expert)
  3. Planner with True Model (MPC using real simulator)
  4. Planner with Learned World Model (MPC using attention model)
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
from training.evaluator import Evaluator


def main():
    # ---- Argument parsing ----
    # Each flag controls a different stage of the pipeline or the environment
    # configuration.  Defaults are tuned for a quick local run; bump
    # --collect-episodes for better trained agents.
    parser = argparse.ArgumentParser(description="Self-Driving Agent Comparison")
    # Road geometry: 1 lane = car-following only; 2+ lanes = lane changes
    parser.add_argument("--num-lanes", type=int, default=2)
    # Number of NPC traffic vehicles sharing the road with the ego
    parser.add_argument("--num-npcs", type=int, default=6)
    # Global random seed for reproducibility across NumPy, PyTorch, and stdlib
    parser.add_argument("--seed", type=int, default=42)
    # How many episodes the expert drives to generate training data (stages 3-4)
    parser.add_argument("--collect-episodes", type=int, default=50,
                        help="Episodes of expert driving to collect")
    # Supervised training epochs for the behavior-cloning policy network
    parser.add_argument("--bc-epochs", type=int, default=200)
    # Training epochs for the attention-based world model
    parser.add_argument("--wm-epochs", type=int, default=20)
    # Number of episodes used to evaluate each agent at the end
    parser.add_argument("--eval-episodes", type=int, default=20)
    # MPC planning horizon (number of future steps the planner considers)
    parser.add_argument("--planner-horizon", type=int, default=30)
    # Number of random action-sequence rollouts the planner samples per step
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = Config()
    cfg.road.num_lanes = args.num_lanes
    cfg.traffic.num_npcs = args.num_npcs
    cfg.sim.episode_steps = 300
    # Select NPC behavior models based on road geometry.
    # behavior_mix is a 3-element probability vector: [random, IDM, IDM+MOBIL].
    # In a single-lane scenario lane changes are impossible, so MOBIL (a
    # lane-change decision model) would be wasted -- use pure IDM instead.
    # With multiple lanes, most NPCs get IDM+MOBIL so they actively change
    # lanes, creating a more realistic and challenging traffic environment.
    if args.num_lanes == 1:
        cfg.traffic.behavior_mix = [0.0, 1.0, 0.0]  # IDM only
    else:
        cfg.traffic.behavior_mix = [0.0, 0.3, 0.7]  # 30% IDM, 70% IDM+MOBIL

    mc = ModelConfig()
    mc.planner_horizon = args.planner_horizon
    mc.planner_num_rollouts = args.planner_rollouts

    # Observation dimensionality:
    #   ego_features = ego state floats (speed, lane_position, lane_id, ...)
    #   + k_neighbors * features_per_neighbor = for each nearby vehicle
    # This flat vector is the input size for the policy and BC networks.
    obs_dim = cfg.obs.ego_features + cfg.obs.k_neighbors * cfg.obs.features_per_neighbor

    print("=" * 60)
    print(f"Self-Driving Simulator Comparison")
    print(f"  Lanes: {args.num_lanes}  NPCs: {args.num_npcs}  Seed: {args.seed}")
    print("=" * 60)

    # ---- 1. Expert Agent ----
    # The expert is a hand-coded rule-based driver.  It serves two purposes:
    #   (a) performance ceiling -- it defines the best behavior we expect, and
    #   (b) teacher -- its trajectories are the training signal for BC and the
    #       world model.
    print("\n[1/5] Expert Agent (rule-based)")
    expert = ExpertAgent(obs_config=cfg.obs, num_lanes=cfg.road.num_lanes)

    # ---- 2. Collect Expert Data ----
    # Roll out the expert agent in the simulator.  The collected transitions
    # (obs, action, reward, next_obs, done) are used for both BC training
    # (obs -> action) and world-model training (obs, action -> next_obs).
    # A dedicated simulator instance (sim_collect) is created so its internal
    # state doesn't interfere with simulators used later for evaluation.
    print(f"\n[2/5] Collecting {args.collect_episodes} episodes of expert data...")
    sim_collect = Simulator(cfg, seed=args.seed)
    collector = DataCollector(sim_collect, expert)
    trajectories = collector.collect_episodes(args.collect_episodes)
    obs_data, act_data, rew_data, nobs_data, done_data = \
        trajectories_to_arrays(trajectories)
    print(f"  Collected {len(obs_data)} transitions "
          f"(avg reward: {np.mean([t.total_reward for t in trajectories]):.1f})")

    # ---- 3. Train Behavior Cloning ----
    # Supervised imitation learning: the policy network is trained to predict
    # the expert's action given the same observation.  BCTrainer internally
    # fits a normalizer (mean/std) on the observation data so the network sees
    # zero-mean, unit-variance inputs.  The normalizer is returned and must be
    # passed to every agent that uses this policy (BC agent).
    print(f"\n[3/5] Training Behavior Cloning ({args.bc_epochs} epochs)...")
    bc_policy = PolicyNetwork(obs_dim, hidden_dim=128, num_layers=2).to(device)
    bc_trainer = BCTrainer(bc_policy, lr=3e-4, batch_size=64)
    bc_losses, bc_normalizer = bc_trainer.train(obs_data, act_data,
                                  num_epochs=args.bc_epochs, verbose=True)
    # deterministic=True makes the agent pick the argmax action (no sampling)
    bc_agent = BCAgent(bc_policy, deterministic=True, normalizer=bc_normalizer)

    # ---- 4. Train World Model ----
    # The world model learns environment dynamics: given (obs, action) predict
    # next_obs.  It uses a transformer-style attention architecture so it can
    # handle a variable number of nearby vehicles.  Once trained, the
    # PlannerLearnedModel agent uses this model as a differentiable simulator
    # to evaluate candidate action sequences via MPC -- avoiding the cost (and
    # the requirement) of querying the true simulator at planning time.
    print(f"\n[4/5] Training Attention World Model ({args.wm_epochs} epochs)...")
    world_model = AttentionWorldModel(
        embed_dim=mc.wm_embed_dim, num_heads=mc.wm_num_heads,
        num_layers=mc.wm_num_layers,
        max_vehicles=mc.wm_max_vehicles, num_actions=9,
        ego_features=cfg.obs.ego_features,
        features_per_neighbor=cfg.obs.features_per_neighbor).to(device)
    wm_trainer = WorldModelTrainer(world_model, lr=3e-4, batch_size=64)
    wm_losses = wm_trainer.train(obs_data, act_data, nobs_data,
                                  num_epochs=args.wm_epochs, verbose=True)

    # ---- Build Planners ----
    # Two MPC-style planners that pick the best action sequence via random
    # shooting over a fixed horizon:
    #
    # planner_true  -- Uses the *real* simulator as the forward model.  This
    #                  gives perfect predictions but is expensive and requires
    #                  simulator access at test time.  It needs its own
    #                  Simulator instance (sim_planner) because it mutates
    #                  simulator state during rollout planning; this same
    #                  instance must also be used during evaluation (see the
    #                  special-case below) so that the planner's internal
    #                  simulator stays in sync with the evaluation environment.
    #
    # planner_learned -- Uses the trained AttentionWorldModel instead of the
    #                    real simulator.  Cheaper at inference and does not
    #                    require access to the true dynamics, but accuracy
    #                    depends on world-model quality.
    print("\nBuilding planners...")
    sim_planner = Simulator(cfg, seed=args.seed + 2)
    planner_true = PlannerTrueModel(sim_planner, mc, seed=args.seed)
    planner_learned = PlannerLearnedModel(world_model, mc, cfg.sim,
                                           cfg.vehicle,
                                           road_config=cfg.road,
                                           normalizer=wm_trainer.normalizer,
                                           obs_config=cfg.obs,
                                           seed=args.seed)

    # ---- 5. Evaluate All Agents ----
    print(f"\n{'='*60}")
    print(f"[5/5] Evaluating all agents ({args.eval_episodes} episodes each)...")
    print(f"{'='*60}")

    agents = {
        "Expert (rule-based)": expert,
        "Behavior Cloning": bc_agent,
        "Planner (true model)": planner_true,
        "Planner (learned WM)": planner_learned,
    }

    # Evaluation loop -- each agent is run for eval_episodes and scored.
    #
    # IMPORTANT: planner_true gets special handling.  Because it plans by
    # calling sim_planner.step() internally (save state -> roll forward ->
    # restore state), the evaluation *must* run on that same sim_planner
    # instance.  If we created a fresh Simulator for evaluation, the planner
    # would still be saving/restoring state on sim_planner while the
    # evaluator advances a *different* simulator, causing the planner's
    # predictions to diverge from reality.
    #
    # All other agents are stateless w.r.t. the simulator, so they can be
    # evaluated on a freshly-constructed Simulator without issue.
    results = {}
    for name, agent in agents.items():
        print(f"\n  Evaluating: {name}...")
        if name == "Planner (true model)":
            # Must reuse the planner's own simulator (see note above)
            evaluator = Evaluator(sim_planner, num_episodes=args.eval_episodes)
        else:
            # Independent simulator so evaluation is isolated from training
            eval_sim = Simulator(cfg, seed=args.seed + 100)
            evaluator = Evaluator(eval_sim, num_episodes=args.eval_episodes)
        results[name] = evaluator.evaluate(agent)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    Evaluator.print_comparison(results)


if __name__ == "__main__":
    main()
