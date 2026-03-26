#!/usr/bin/env python3
"""Record one episode per agent and save frame data as JSON for visualization.

Produces a JSON file containing an array of replay objects (one per agent).
Each replay holds the road geometry, episode summary stats, and a list of
per-timestep "frames" that capture the full simulation state.  The output
file is designed to be loaded by debug_viewer.html, which renders the frames
as an interactive step-by-step animation of the highway environment.

JSON structure (top-level array):
  [
    {
      "agent": "<name>",
      "num_lanes": int,
      "lane_width": float,
      "total_reward": float,
      "steps": int,
      "collision": bool,
      "frames": [ { step, reward, collision, ego, npcs, action }, ... ]
    },
    ...
  ]

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
    """Record an episode as a list of frames.

    Each frame is a snapshot of the full simulation state at one timestep,
    including the ego vehicle position/speed, all NPC positions/speeds, the
    action taken, the reward received, and whether a collision occurred.
    Frame 0 captures the initial state before any action is taken (action=None).

    Returns a dict summarising the episode (agent name, road geometry, reward,
    collision flag) along with the full list of frames.
    """
    obs = sim.reset()
    agent.reset()
    frames = []
    done = False
    step = 0
    total_reward = 0.0

    # Record the initial state (step 0) before the agent acts.
    # action is None because no decision has been made yet.
    ego, npcs = sim.get_all_vehicle_states()
    frames.append(_make_frame(step, ego, npcs, None, 0.0, False))

    # Step the simulation until the episode terminates or we hit max_steps.
    # Each iteration: agent chooses an action from the observation, the sim
    # advances one tick, and we snapshot the resulting state into a new frame.
    while not done and step < max_steps:
        action = agent.act(obs)
        obs, reward, done, info = sim.step(action)
        step += 1
        total_reward += reward
        ego, npcs = sim.get_all_vehicle_states()
        frames.append(_make_frame(step, ego, npcs, action, reward,
                                   info.get("collision", False)))

    # Episode-level summary that wraps the frame list.
    return {
        "agent": agent_name,           # display name for the viewer
        "num_lanes": sim.road.num_lanes,
        "lane_width": sim.road.lane_width,
        "total_reward": round(total_reward, 1),
        "steps": step,
        "collision": info.get("collision", False),
        "frames": frames,
    }


def _make_frame(step, ego, npcs, action, reward, collision):
    """Build a single frame dict from raw simulation state.

    Fields:
      step      - integer timestep index (0 = initial state)
      reward    - float reward earned on this transition (0.0 for step 0)
      collision - True if the ego vehicle collided this step
      ego       - dict with ego x position (metres), lane index, speed (m/s)
      npcs      - list of dicts, one per NPC, with vehicle_id, x, lane, speed
      action    - (optional) dict with longitudinal and lateral action enums
                  cast to ints; omitted for the initial frame (step 0)
    """
    frame = {
        "step": step,
        "reward": round(reward, 3),
        "collision": collision,
        # Ego vehicle state: longitudinal position, current lane, current speed
        "ego": {
            "x": round(ego.x, 2),
            "lane": ego.lane,
            "speed": round(ego.speed, 2),
        },
        # All NPC vehicles visible in the simulation at this timestep
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
    # The initial frame (step 0) has no action; only subsequent frames include
    # the longitudinal (accel/maintain/brake) and lateral (left/stay/right)
    # action that was applied to reach this state.
    if action is not None:
        frame["action"] = {
            "lon": int(action.longitudinal),
            "lat": int(action.lateral),
        }
    return frame


def main():
    # --- CLI argument parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-lanes", type=int, default=2)
    parser.add_argument("--num-npcs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="replay_data.json")
    args = parser.parse_args()

    # --- Build simulation config from CLI args ---
    cfg = Config()
    cfg.road.num_lanes = args.num_lanes
    cfg.traffic.num_npcs = args.num_npcs
    cfg.sim.episode_steps = 300

    # Register the agents to record.  Currently only the rule-based expert is
    # included because it has no torch dependency.  Additional learned agents
    # (BC, RL, planners) can be added here once torch is available.
    agents = {}
    agents["Expert"] = ExpertAgent(
        obs_config=cfg.obs, num_lanes=cfg.road.num_lanes)

    # --- Record one episode per agent using the same seed for fair comparison ---
    replays = []
    for name, agent in agents.items():
        print(f"Recording {name}...")
        sim = Simulator(cfg, seed=args.seed)
        replay = record_episode(sim, agent, name)
        replays.append(replay)
        print(f"  {replay['steps']} steps, reward={replay['total_reward']}, "
              f"collision={replay['collision']}")

    # --- Write the replay list to a JSON file for debug_viewer.html ---
    with open(args.output, "w") as f:
        json.dump(replays, f)
    print(f"\nSaved {len(replays)} replays to {args.output}")


if __name__ == "__main__":
    main()
