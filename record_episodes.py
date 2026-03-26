#!/usr/bin/env python3
"""Record one episode per agent and save frame data as JSON for visualization.

Usage:
    python record_episodes.py [--num-lanes 2] [--seed 42] [--output replay_data.json]
"""
import argparse
import json
import sys
sys.path.insert(0, ".")

from utils.config import Config
from utils.types import Action
from environment.simulator import Simulator
from agents.expert import ExpertAgent


def record_episode(sim: Simulator, agent, agent_name: str,
                   max_steps: int = 300) -> dict:
    """Record an episode as a list of frames."""
    obs = sim.reset()
    agent.reset()
    frames = []
    done = False
    step = 0
    total_reward = 0.0

    # Record initial frame
    ego, npcs = sim.get_all_vehicle_states()
    frames.append(_make_frame(step, ego, npcs, None, 0.0, False))

    while not done and step < max_steps:
        action = agent.act(obs)
        obs, reward, done, info = sim.step(action)
        step += 1
        total_reward += reward
        ego, npcs = sim.get_all_vehicle_states()
        frames.append(_make_frame(step, ego, npcs, action, reward,
                                   info.get("collision", False)))

    return {
        "agent": agent_name,
        "num_lanes": sim.road.num_lanes,
        "lane_width": sim.road.lane_width,
        "total_reward": round(total_reward, 1),
        "steps": step,
        "collision": info.get("collision", False),
        "frames": frames,
    }


def _make_frame(step, ego, npcs, action, reward, collision):
    frame = {
        "step": step,
        "reward": round(reward, 3),
        "collision": collision,
        "ego": {
            "x": round(ego.x, 2),
            "lane": ego.lane,
            "speed": round(ego.speed, 2),
        },
        "npcs": [
            {
                "x": round(n.x, 2),
                "lane": n.lane,
                "speed": round(n.speed, 2),
                "id": n.vehicle_id,
            }
            for n in npcs
        ],
    }
    if action is not None:
        frame["action"] = {
            "lon": int(action.longitudinal),
            "lat": int(action.lateral),
        }
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-lanes", type=int, default=2)
    parser.add_argument("--num-npcs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="replay_data.json")
    args = parser.parse_args()

    cfg = Config()
    cfg.road.num_lanes = args.num_lanes
    cfg.traffic.num_npcs = args.num_npcs
    cfg.sim.episode_steps = 300

    # For now, record just the expert (no torch dependency)
    # When torch is available, the full comparison can be recorded
    agents = {}
    agents["Expert"] = ExpertAgent(
        obs_config=cfg.obs, num_lanes=cfg.road.num_lanes)

    replays = []
    for name, agent in agents.items():
        print(f"Recording {name}...")
        sim = Simulator(cfg, seed=args.seed)
        replay = record_episode(sim, agent, name)
        replays.append(replay)
        print(f"  {replay['steps']} steps, reward={replay['total_reward']}, "
              f"collision={replay['collision']}")

    with open(args.output, "w") as f:
        json.dump(replays, f)
    print(f"\nSaved {len(replays)} replays to {args.output}")


if __name__ == "__main__":
    main()
