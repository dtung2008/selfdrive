#!/usr/bin/env python3
"""
Record a detailed episode for visual debugging.

This is an enhanced version of record_episodes.py that captures additional
per-frame diagnostic data for every agent type.  On top of the basic state
snapshot (ego, NPCs, reward, collision), each frame also includes:

  - expert_action : the action the rule-based expert *would* have taken at
    this observation, enabling side-by-side comparison with the tested agent.
  - lane_gaps     : for every lane, the closest NPC ahead and behind the ego
    vehicle (in metres).  Useful for understanding gap-selection behaviour.
  - disagree      : boolean flag that is True when the agent's chosen action
    differs from the expert's, making it easy to spot divergence in the viewer.
  - gap_ahead / gap_behind : current-lane gaps kept for backward compatibility.
  - cumulative_reward : running reward total so the viewer can plot a reward
    curve alongside the animation.

The output JSON wraps the frame list with a version tag, full config, training
metadata (agent_info), and an episode summary.  It is consumed by
debug_viewer.html for interactive step-through visualisation.

Supports all agent types: expert, bc, rl, planner_true, planner_wm.  Learned
agents (bc, rl, planner_wm) are trained from scratch each run using expert
demonstration data, so the output is fully self-contained and reproducible
given the same seeds.

Usage:
    python record_debug.py --agent expert --seed 42 --num-lanes 2 --output debug_episode.json
    python record_debug.py --agent bc --seed 0 --num-lanes 2 --bc-epochs 50
    python record_debug.py --agent planner_true --planner-horizon 30
    python record_debug.py --agent planner_wm --wm-epochs 80
    python record_debug.py --agent rl --rl-episodes 300
"""
import argparse
import json
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
    """Build a Config with CLI overrides applied.

    Overrides road geometry (num_lanes, num_npcs, episode length) from args.
    Also adjusts the NPC traffic behaviour mix based on lane count:
      - 1 lane:  all NPCs use the "medium" behaviour (no lane changes possible)
      - 2+ lanes: mostly "fast" NPCs with some "medium", creating overtaking
        scenarios that stress-test lane-change decisions.
    """
    cfg = Config()
    cfg.road.num_lanes = args.num_lanes
    cfg.traffic.num_npcs = args.num_npcs
    cfg.sim.episode_steps = args.max_steps
    if args.num_lanes == 1:
        # Single lane: no aggressive NPCs (lane changes impossible)
        cfg.traffic.behavior_mix = [0.0, 1.0, 0.0]
    else:
        # Multi-lane: heavier weight on fast NPCs to create interesting gaps
        cfg.traffic.behavior_mix = [0.0, 0.3, 0.7]
    return cfg


def build_agent(args, cfg, obs_dim, device=None):
    """Build the requested agent, training if necessary.

    For the expert agent no training is needed -- it uses hand-crafted rules.
    All learned agents first collect expert demonstration data, then train
    their respective models before returning a ready-to-evaluate agent.

    Returns (agent, info) where info is a dict of training metadata that
    gets written into the output JSON for provenance.
    """
    expert = ExpertAgent(obs_config=cfg.obs, num_lanes=cfg.road.num_lanes)
    info = {"agent_type": args.agent}

    # --- Expert: rule-based, no training required ---
    if args.agent == "expert":
        return expert, info

    # --- Collect expert demonstrations (shared by all learned agents) ---
    # The expert drives the simulator for N episodes; the resulting
    # (obs, action, reward, next_obs, done) tuples form the training set.
    print(f"Collecting {args.collect_episodes} expert episodes...")
    sim_c = Simulator(cfg, seed=args.train_seed)
    collector = DataCollector(sim_c, expert)
    trajs = collector.collect_episodes(args.collect_episodes)
    obs_data, act_data, rew_data, nobs_data, done_data = trajectories_to_arrays(trajs)
    info["train_transitions"] = len(obs_data)

    # --- Behavioural Cloning (BC): supervised learning on expert actions ---
    # Trains a policy network to predict the expert's action from each
    # observation using cross-entropy loss.  The resulting agent is wrapped
    # in BCAgent with deterministic=True (argmax, no sampling).
    if args.agent == "bc":
        print(f"Training BC ({args.bc_epochs} epochs)...")
        policy = PolicyNetwork(obs_dim, hidden_dim=128, num_layers=2).to(device)
        trainer = BCTrainer(policy, lr=3e-4, batch_size=64)
        losses, normalizer = trainer.train(obs_data, act_data, num_epochs=args.bc_epochs, verbose=True)
        info["bc_final_loss"] = losses[-1]
        return BCAgent(policy, deterministic=True, normalizer=normalizer), info

    # --- Reinforcement Learning (RL): BC warm-start then policy gradient ---
    # Two-phase training pipeline:
    #   1. Pre-train a policy via BC so it starts from a reasonable baseline.
    #   2. Fine-tune with REINFORCE (policy gradient) using environment reward,
    #      reusing the BC observation normalizer for stable inputs.
    # The final policy is evaluated deterministically through BCAgent.
    if args.agent == "rl":
        print(f"Training BC for warm-start ({args.bc_epochs} epochs)...")
        bc_policy = PolicyNetwork(obs_dim, hidden_dim=128, num_layers=2).to(device)
        bc_trainer = BCTrainer(bc_policy, lr=3e-4, batch_size=64)
        bc_trainer.train(obs_data, act_data, num_epochs=args.bc_epochs, verbose=False)
        bc_normalizer = bc_trainer.normalizer

        print(f"Training RL ({args.rl_episodes} episodes, warm-start)...")
        rl_policy = copy.deepcopy(bc_policy)
        sim_rl = Simulator(cfg, seed=args.train_seed + 1)
        rl_trainer = RLTrainer(rl_policy, sim_rl, lr=1e-4, gamma=0.99, entropy_coef=0.005)
        # Use BC normalizer instead of building a new one
        rl_trainer.obs_normalizer = bc_normalizer
        rewards = rl_trainer.train(num_episodes=args.rl_episodes, verbose=True)
        info["rl_final_avg_reward"] = float(np.mean(rewards[-50:]))
        return BCAgent(rl_policy, deterministic=True, normalizer=bc_normalizer), info

    # --- Planner with true model: random-shooting MPC using the real sim ---
    # Uses the actual simulator as a perfect forward model.  At each step it
    # generates random action sequences, rolls them out in a cloned sim, and
    # picks the sequence with the highest cumulative reward.
    if args.agent == "planner_true":
        mc = ModelConfig()
        mc.planner_horizon = args.planner_horizon
        mc.planner_num_rollouts = args.planner_rollouts
        sim_p = Simulator(cfg, seed=args.seed)
        info["planner_horizon"] = mc.planner_horizon
        info["planner_rollouts"] = mc.planner_num_rollouts
        return PlannerTrueModel(sim_p, mc, seed=args.seed), info

    # --- Planner with learned world model: MPC using an attention-based WM ---
    # First trains a transformer world model to predict next observations from
    # (obs, action) pairs.  Then uses that learned model as the forward model
    # inside the same random-shooting planner.  planner_wm_horizon may be
    # shorter than planner_horizon because learned-model errors compound.
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
                                  features_per_neighbor=cfg.obs.features_per_neighbor).to(device)
        wm_trainer = WorldModelTrainer(wm, lr=3e-4, batch_size=64)
        wm_losses = wm_trainer.train(obs_data, act_data, nobs_data,
                                      num_epochs=args.wm_epochs, verbose=True)
        info["wm_final_loss"] = wm_losses[-1]
        info["planner_wm_horizon"] = mc.planner_wm_horizon
        return PlannerLearnedModel(wm, mc, cfg.sim, cfg.vehicle,
                                    road_config=cfg.road,
                                    normalizer=wm_trainer.normalizer,
                                    obs_config=cfg.obs,
                                    seed=args.seed), info

    raise ValueError(f"Unknown agent: {args.agent}")


def record_episode(sim, agent, expert, max_steps):
    """Record one episode with detailed per-step data.

    In addition to the basic simulation state, each frame captures debug data
    that the viewer uses for analysis overlays:

      expert_action - what the expert would have done at the same observation,
                      letting the viewer highlight where the agent diverges.
      lane_gaps     - per-lane dict mapping lane index to {ahead, behind}
                      distances (metres) to the nearest NPC.  None means no
                      NPC in that direction.  Useful for understanding why the
                      expert chose a particular lane change.
      disagree      - boolean shorthand: True when agent and expert chose
                      different discrete actions (compared via to_index()).
      gap_ahead / gap_behind - ego-lane gaps, kept for backward compat.
      cumulative_reward - running total so the viewer can plot reward curves.

    Returns (frames, info) where info is the final step's info dict from the
    simulator (used to extract collision status).
    """
    obs = sim.reset()
    agent.reset()
    frames = []
    done = False
    step = 0
    cumulative_reward = 0.0

    # Initial frame (step 0): no actions have been taken yet, so action and
    # expert_action are both None and reward/cumulative_reward are zero.
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
        # Query both the agent under test and the expert on the *same*
        # observation so their actions are directly comparable.
        action = agent.act(obs)
        expert_action = expert.act(obs)
        obs, reward, done, info = sim.step(action)
        step += 1
        cumulative_reward += reward

        ego, npcs = sim.get_all_vehicle_states()
        num_lanes = sim.road.num_lanes

        # Compute the distance to the nearest NPC ahead and behind the ego
        # vehicle in every lane.  This gives the viewer a full picture of
        # available gaps, not just the ego's current lane.
        lane_gaps = {}
        for lane in range(num_lanes):
            ahead_in_lane = [n for n in npcs if n.lane == lane and n.x > ego.x]
            behind_in_lane = [n for n in npcs if n.lane == lane and n.x < ego.x]
            lane_gaps[lane] = {
                "ahead": round(min(n.x - ego.x for n in ahead_in_lane), 1) if ahead_in_lane else None,
                "behind": round(min(ego.x - n.x for n in behind_in_lane), 1) if behind_in_lane else None,
            }

        # Current-lane gaps extracted separately for backward compatibility
        # with older viewer versions that don't read lane_gaps.
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
            # Flag steps where the agent and expert disagree, so the viewer
            # can highlight these frames (e.g. with a red border).
            "disagree": action.to_index() != expert_action.to_index(),
        })

    return frames, info


def main():
    # --- CLI arguments ---
    # Two seed arguments: --seed controls the episode being recorded, while
    # --train-seed controls the expert data collection and model training.
    # Keeping them separate lets you train once and evaluate on many episodes.
    parser = argparse.ArgumentParser(description="Record debug episode")
    parser.add_argument("--agent", type=str, default="expert",
                        choices=["expert", "bc", "planner_true", "planner_wm", "rl"])
    parser.add_argument("--seed", type=int, default=42, help="Episode seed")
    parser.add_argument("--train-seed", type=int, default=42, help="Training data seed")
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
    parser.add_argument("--output", type=str, default="debug_episode.json")
    args = parser.parse_args()

    cfg = make_config(args)
    # Observation dimension: ego_features plus features_per_neighbor for
    # each of the k nearest NPCs.
    obs_dim = cfg.obs.ego_features + cfg.obs.k_neighbors * cfg.obs.features_per_neighbor

    # Seed all RNGs (Python, NumPy, PyTorch CPU+CUDA) for full reproducibility
    # of training and evaluation.
    import random
    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build the agent (may involve training) and a separate expert instance
    # that will be queried in parallel during recording for comparison.
    agent, agent_info = build_agent(args, cfg, obs_dim, device)
    expert = ExpertAgent(obs_config=cfg.obs, num_lanes=cfg.road.num_lanes)

    # --- Record the episode ---
    print(f"\nRecording episode: agent={args.agent} seed={args.seed} "
          f"lanes={args.num_lanes} npcs={args.num_npcs}")

    # The planner_true agent internally mutates its own simulator (for
    # look-ahead rollouts), so we must use that same sim for recording.
    # All other agents get a fresh simulator seeded for the evaluation episode.
    if args.agent == "planner_true":
        sim = agent.sim  # planner needs its own sim
    else:
        sim = Simulator(cfg, seed=args.seed)

    frames, info = record_episode(sim, agent, expert, args.max_steps)

    # --- Assemble the output JSON ---
    # Top-level structure consumed by debug_viewer.html:
    #   version    : schema version (2) so the viewer can handle format changes
    #   config     : environment and episode parameters for display/filtering
    #   agent_info : training metadata (type, losses, reward curves, etc.)
    #   summary    : quick stats (steps, reward, collision, disagree count)
    #   frames     : the full per-step data array (see record_episode docstring)
    output = {
        "version": 2,
        "config": {
            "agent": args.agent,
            "seed": args.seed,
            "num_lanes": cfg.road.num_lanes,
            "num_npcs": cfg.traffic.num_npcs,
            "lane_width": cfg.road.lane_width,
            "max_steps": args.max_steps,
            "dt": cfg.sim.dt,
        },
        "agent_info": agent_info,
        "summary": {
            "total_steps": len(frames) - 1,  # exclude the initial frame
            "total_reward": frames[-1]["cumulative_reward"] if frames else 0,
            "collision": info.get("collision", False),
            "final_speed": frames[-1]["ego"]["speed"] if frames else 0,
            # Count of frames where the agent chose differently from the expert
            "disagree_count": sum(1 for f in frames if f.get("disagree", False)),
        },
        "frames": frames,
    }

    # Write with indent=2 for human readability (files are typically < 1 MB)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    col = "COLLISION" if info.get("collision") else "OK"
    print(f"\n[{col}] {len(frames)-1} steps, reward={output['summary']['total_reward']:.1f}, "
          f"disagrees={output['summary']['disagree_count']}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
