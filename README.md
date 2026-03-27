# Self-Driving Simulator

A modular simulator for learning and comparing self-driving control strategies on a multi-lane highway. The project implements a complete pipeline from environment simulation through expert data collection, model training, and multi-agent evaluation.

## Overview

This project provides:

- A **multi-lane highway simulator** with realistic traffic (IDM car-following, MOBIL lane-changing)
- Five **driving agents** spanning rule-based, supervised learning, model-predictive control, and reinforcement learning
- An **attention-based world model** (Transformer) that learns to predict traffic dynamics
- A **training pipeline** for behavior cloning, world model learning, and policy gradient RL
- An **evaluation harness** for side-by-side agent comparison with metrics like collision rate, average speed, and episode completion

## Agents Compared

| Agent | Type | Description |
|---|---|---|
| **Expert** | Rule-based | Acceleration-aware longitudinal control with safe lane-change decisions, urgency escalation, and configurable parameters via `ExpertConfig`. Used to generate demonstration data for supervised learning. |
| **Behavior Cloning (BC)** | Supervised | MLP policy trained to imitate the expert via cross-entropy loss with inverse-frequency class weighting for rare actions. |
| **Planner (true model)** | MPC | Random-shooting model-predictive control using the real simulator's `clone_state`/`restore_state` for exact rollouts. Evaluates 9 maneuver templates over a configurable horizon. Uses split RNGs for candidate generation vs. tie-breaking. |
| **Planner (learned WM)** | MPC | Same MPC framework but replaces the simulator with a learned attention-based world model. Runs rollouts entirely in PyTorch for speed. |
| **RL (REINFORCE)** | Policy gradient | MLP policy trained via REINFORCE with running baseline, advantage whitening, entropy bonus, and gradient clipping. |

## Traffic Models

NPCs use realistic car-following and lane-change models from the traffic simulation literature:

- **Constant Speed** -- Baseline vehicles maintaining a fixed target speed with proportional control.
- **IDM (Intelligent Driver Model)** -- Physics-based car-following (Treiber et al., 2000) with configurable desired speed `v0`, time headway `T`, minimum gap `s0`, and comfort acceleration/braking `a`/`b`. Produces smooth, collision-free longitudinal behavior.
- **MOBIL (Minimizing Overall Braking Induced by Lane changes)** -- Incentive-based lane-change model (Kesting et al., 2007) layered on top of IDM. Checks a safety criterion (follower deceleration < threshold) and an incentive criterion (net acceleration gain > politeness-weighted cost + threshold).
- **Personality Variants** -- Gaussian noise applied to IDM/MOBIL parameters creates a mix of aggressive and timid drivers, improving training distribution diversity.

## Architecture

### Action Space

9 discrete actions arranged as a 3x3 grid:

```
                  Lateral
              LEFT  KEEP  RIGHT
           +------+------+------+
DECELERATE |  0   |  1   |  2   |
Lon  KEEP  |  3   |  4   |  5   |
ACCELERATE |  6   |  7   |  8   |
           +------+------+------+

Index formula: (longitudinal + 1) * 3 + (lateral + 1)
```

### Observation Space

Fixed-size vector with ego state followed by K nearest neighbors (default K=10):

```
[ego_speed, ego_lane, ego_x, ego_accel,         # 4 values: ego state
 rel_x_1, rel_lane_1, rel_speed_1, rel_accel_1, exists_1,   # 5 values per neighbor
 rel_x_2, rel_lane_2, rel_speed_2, rel_accel_2, exists_2,   # sorted by distance
 ...                                                          # zero-padded if < K
 rel_x_K, rel_lane_K, rel_speed_K, rel_accel_K, exists_K]   # total: 4 + 5*K = 54
```

### Attention World Model

The world model uses a Transformer encoder to predict next-state deltas:

1. **Per-vehicle embedding** -- Each vehicle's features are projected to `embed_dim` via separate MLPs for ego and neighbors
2. **Action injection** -- One-hot ego action is concatenated before embedding
3. **Type embedding** -- Learned tokens distinguish ego (position 0) from NPC vehicles
4. **Transformer encoder** -- Multi-head self-attention captures inter-vehicle interactions (e.g., braking cascades). Attention masking handles variable neighbor counts
5. **Residual prediction** -- Output heads predict state *deltas* rather than absolute values, making learning easier when consecutive states are similar

## Project Structure

```
selfdrive/
├── environment/                     # Simulation engine
│   ├── road.py                      # Road geometry: N lanes, lane widths, bounds checking
│   ├── vehicle.py                   # First-order kinematic dynamics (Euler integration)
│   ├── traffic_behaviors.py         # NPC behaviors: Constant Speed, IDM, MOBIL
│   ├── traffic_manager.py           # NPC lifecycle: spawn, step, despawn, respawn
│   ├── observation.py               # Ego-centric fixed-size observation builder
│   └── simulator.py                 # Gym-like step/reset with reward computation
├── agents/                          # Driving agents
│   ├── base.py                      # Abstract Agent interface: act(obs) -> Action
│   ├── expert.py                    # Rule-based expert with IDM-like logic
│   ├── bc_agent.py                  # Behavior cloning agent with safety override
│   ├── planner_true.py              # MPC planner using true simulator rollouts
│   └── planner_learned.py           # MPC planner using learned world model
├── models/                          # Neural network architectures
│   ├── policy_net.py                # MLP policy: obs -> action logits (9 classes)
│   └── world_model.py              # Transformer world model: (obs, action) -> next_obs
├── training/                        # Training loops and data collection
│   ├── data_collector.py            # Trajectory collection via agent-environment interaction
│   ├── bc_trainer.py                # Behavior cloning: cross-entropy + class weighting
│   ├── world_model_trainer.py       # World model: MSE on normalized observation deltas
│   ├── rl_trainer.py                # REINFORCE with baseline, entropy, grad clipping
│   └── evaluator.py                 # Multi-agent evaluation with statistical metrics
├── utils/                           # Shared types and configuration
│   ├── types.py                     # Core dataclasses: Action, VehicleState, Observation,
│   │                                #   Transition, Trajectory
│   └── config.py                    # All hyperparameters: road, vehicle, traffic, sim,
│                                    #   observation, model, expert configs
├── tests/                           # 103 unit tests covering all modules
│   ├── test_types_config.py         # Action encoding, VehicleState, Observation, Config
│   ├── test_road_vehicle.py         # Road geometry, VehicleDynamics step functions
│   ├── test_traffic_behaviors.py    # IDM/MOBIL/ConstantSpeed + helper functions
│   ├── test_traffic_manager.py      # NPC spawn/despawn, gap enforcement, multi-lane
│   ├── test_observation.py          # Observation shape, sorting, padding, truncation
│   ├── test_simulator.py            # Step/reset, collision, reward, clone/restore
│   ├── test_expert.py               # Expert decisions on crafted observations
│   ├── test_models.py               # PolicyNetwork + AttentionWorldModel shapes/gradients
│   ├── test_training.py             # DataCollector, BCTrainer, WorldModelTrainer
│   └── test_planners_eval.py        # PlannerTrue, PlannerLearned, Evaluator
├── run_comparison.py                # Full pipeline: collect data, train all agents, compare
├── run_simple_debug.py              # Simplified 1-lane comparison for isolating longitudinal control
├── record_episodes.py               # Record episode replays as JSON for browser visualization
├── record_debug.py                  # Enhanced debug recording with expert comparison per step
├── trace_bc_collision.py            # Step-by-step BC collision diagnosis (BC vs Expert)
├── test_wm_horizon.py              # WM planner error-compounding test across horizons
├── verify_wm_rollout.py            # Hybrid WM rollout accuracy vs true simulator
├── debug_planner_decision.py        # Planner reward breakdown at stuck/slow steps
├── scan_failures.py                 # Scan seed ranges for agent failure episodes
├── debug_viewer.html                # Browser-based episode visualization (loads JSON replays)
├── requirements.txt                 # Dependencies: numpy, torch, pytest
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- `numpy >= 1.21` -- Array operations and linear algebra
- `torch >= 2.0` -- Neural networks and autograd (CUDA used automatically when available)
- `pytest >= 7.0` -- Test framework

## Running Tests

Run all 103 tests:
```bash
python -m pytest tests/ -v
```

Run a specific test file:
```bash
python -m pytest tests/test_traffic_behaviors.py -v
python -m pytest tests/test_simulator.py -v
```

Run a single test:
```bash
python -m pytest tests/test_expert.py::TestExpertAgent::test_accelerates_on_empty_road -v
```

## Running the Full Comparison

Train all agents and evaluate them side by side:

```bash
python run_comparison.py
```

### Command-Line Options

```bash
python run_comparison.py \
  --num-lanes 2 \              # Number of highway lanes (default: 2)
  --num-npcs 6 \               # Number of NPC vehicles (default: 6)
  --collect-episodes 50 \      # Expert episodes for training data (default: 50)
  --bc-epochs 200 \             # Behavior cloning training epochs (default: 200)
  --wm-epochs 20 \             # World model training epochs (default: 20)
  --rl-episodes 300 \          # REINFORCE training episodes (default: 300)
  --eval-episodes 30           # Evaluation episodes per agent (default: 20)
```

### Pipeline Steps

1. **Data Collection** -- Expert agent drives in the simulator, collecting (observation, action) trajectories
2. **BC Training** -- Train an MLP policy to imitate expert actions via supervised learning
3. **World Model Training** -- Train the attention world model to predict next observations
4. **RL Training** -- Train a separate MLP policy via REINFORCE in the live simulator
5. **Evaluation** -- Run all agents for N episodes, collecting metrics:
   - Average reward and standard deviation
   - Average speed (m/s)
   - Collision rate
   - Average episode length
   - Lane changes per episode
   - Episode completion rate

## Debugging and Diagnostic Tools

The project includes several scripts for diagnosing agent behavior and world model accuracy.

### Simplified 1-Lane Debug

Isolate longitudinal control by removing lane changes entirely. All agents should achieve 0% collision rate in this setting:

```bash
python run_simple_debug.py --seed 42 --eval-episodes 20
```

### Record and Visualize Episodes

Record an episode as JSON, then open `debug_viewer.html` in a browser to visualize it:

```bash
# Record expert agent replay
python record_episodes.py --num-lanes 2 --output replay_data.json

# Record any agent with enhanced debug data (includes expert comparison)
python record_debug.py --agent bc --seed 42 --output debug_episode.json
python record_debug.py --agent planner_wm --wm-epochs 80 --output debug_episode.json
python record_debug.py --agent rl --rl-episodes 300 --output debug_episode.json
```

Then open `debug_viewer.html` in a browser and load the JSON file.

### Diagnose BC Collisions

Find a seed where the BC agent collides, then replay step-by-step comparing BC vs Expert decisions:

```bash
python trace_bc_collision.py
```

Output shows per-step speed, gaps, and where BC disagrees with the expert (the likely collision cause).

### World Model Accuracy

Test whether WM prediction errors compound over longer planning horizons:

```bash
# Collision rate vs planning horizon (expect spike at longer horizons)
python test_wm_horizon.py

# Step-by-step comparison: hybrid WM rollout vs true simulator
python verify_wm_rollout.py
```

### Scan for Failure Seeds

Train an agent once, then run it across a range of seeds. Only outputs debug JSON for episodes that end early (collisions or other early terminations), skipping full-length episodes silently. Useful for hunting down rare failure modes without manually trying seeds one at a time:

```bash
# Scan seeds 0-100 for planner_wm failures
python scan_failures.py --agent planner_wm --seeds 0-100

# Scan with specific config
python scan_failures.py --agent bc --seeds 0-50 --num-lanes 1

# Save failure JSON files to a directory
python scan_failures.py --agent planner_wm --seeds 0-200 --wm-epochs 80 --output-dir failures/
```

Output includes a summary of failure rate and per-failure JSON files compatible with `debug_viewer.html`.

### Planner Decision Debugging

Dump the WM planner's internal reward breakdown when it gets "stuck" (speed < 22 m/s for 5+ steps):

```bash
python debug_planner_decision.py --seed 42 --debug-steps stuck
python debug_planner_decision.py --seed 42 --debug-steps 50,100,150
```

## Performance Comparison

Average reward across lane configurations (3 seeds each, 300 steps, planners use h=30, r=50):

| Config | Expert | BC | RL | Planner True | Planner WM |
|-----------|--------|--------|--------|--------------|------------|
| 1L / 3NPC | 200.4 | 198.9 | 199.3 | 203.5 | 198.9 |
| 2L / 6NPC | 117.4 | 215.5 | 172.9 | 239.7 | 246.9 |
| 3L / 9NPC | 281.9 | 204.5 | 178.3 | 244.4 | 231.8 |
| 4L / 12NPC | 204.9 | 189.6 | 158.9 | 242.2 | 237.5 |

Key observations:
- **1L**: All agents perform similarly (~200). Simple car-following is well-handled by all.
- **2L**: Expert is volatile across seeds (33–271) while BC and both planners are consistent. Planner WM slightly outperforms planner true here (247 vs 240).
- **3L/4L**: Expert excels when its heuristics find good lanes (282–287) but can collapse on hard seeds (46). Planners are the most consistent (~240), while BC plateaus around 200.
- **BC**: Zero collisions across all configs thanks to the safety override layer, but the MLP can't replicate multi-step lane-change reasoning, capping reward around 200.
- **RL**: Performs worse than BC in multi-lane settings. REINFORCE fine-tuning with 300 episodes degrades the BC-initialized policy rather than improving it — high variance gradients cause policy drift without learning better strategies.
- **Planner WM vs True**: Remarkably close despite using a learned model trained for only 20 epochs.

## Key Design Decisions

### Modularity
Every module has a clean interface and its own test file. Adding a new agent requires:
1. Subclass `Agent` in `agents/` -- implement `act(obs: np.ndarray) -> Action`
2. Add it to the comparison dict in `run_comparison.py`
3. Write tests in `tests/`

### Extensibility
- **More lanes** -- Change `num_lanes` in config; all modules handle N-lane roads
- **Continuous actions** -- Modify `Action` class and `VehicleDynamics`; agents and models adapt
- **Better RL** -- Swap `rl_trainer.py` internals (e.g., PPO); the `Agent` interface stays the same
- **Richer world model** -- Add recurrence or graph attention to `world_model.py`; planner interface unchanged
- **New traffic behaviors** -- Subclass `TrafficBehavior` and add to the behavior mix in `TrafficConfig`

### CUDA Support

All neural network models (PolicyNetwork, AttentionWorldModel) are automatically placed on CUDA when a GPU is available. Both `record_debug.py` and `run_comparison.py` detect the device at startup and pass it through to model creation. Trainers (`BCTrainer`, `WorldModelTrainer`, `RLTrainer`) and inference agents (`BCAgent`, `PlannerLearnedModel`) derive the device from model parameters, so tensors are moved automatically.

### Expert Configuration

The expert agent's behavior is fully parameterized via `ExpertConfig` in `utils/config.py`. Key parameters include:

- **Gap thresholds**: `normal_min_gap_ahead/behind`, `emergency_gap_ahead/behind`, `critical_gap`
- **Speed control**: `desired_speed`, `speed_deadband_low/high`, `max_decel`
- **Lane changes**: `lane_change_gap`, `lane_change_cooldown`, `min_improvement`, `closing_speed_threshold`
- **Urgency escalation**: `urgency_escalation_steps`, `max_urgency_escalation` — only relaxes the ahead gap (never the behind gap, which is a safety constraint)
- **Lookahead**: `proactive_lane_change_time`, `two_step_lookahead_factor`, `gap_projection_time`
- **Cascade detection**: `cascade_accel_threshold`, `cascade_max_gap`

### Reward Structure

The reward function in the simulator combines multiple terms:
- **Collision penalty** -- Large negative reward on collision (default: -100)
- **Speed reward** -- Linear reward for driving near the speed limit
- **Lane-change penalty** -- Small penalty to discourage unnecessary lane changes
- **Hard-brake penalty** -- Penalizes abrupt deceleration for ride comfort
