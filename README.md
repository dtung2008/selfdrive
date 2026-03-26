# Self-Driving Simulator

A modular simulator for learning and comparing self-driving control strategies on a multi-lane highway.

## Agents Compared

| Agent | Type | Description |
|---|---|---|
| **Expert** | Rule-based | IDM-like speed control + safe lane changes. Generates training data. |
| **Behavior Cloning** | Supervised | MLP policy trained to imitate the expert. |
| **Planner (true model)** | MPC | Random-shooting MPC using the real simulator for rollouts. |
| **Planner (learned WM)** | MPC | Same MPC but rollouts use an attention-based world model. |
| **RL (REINFORCE)** | Policy gradient | MLP policy trained via REINFORCE with baseline. |

## Traffic Models

NPCs use realistic car-following and lane-change models:

- **Constant Speed** — baseline vehicles maintaining fixed speed
- **IDM (Intelligent Driver Model)** — physics-based car-following with configurable desired speed, time headway, and comfort braking
- **MOBIL** — incentive-based lane changes on top of IDM, checking safety constraints
- **Personality Variants** — randomized IDM/MOBIL parameters create aggressive/timid driver mixes

## Project Structure

```
selfdrive/
├── environment/
│   ├── road.py               # Road geometry (N lanes)
│   ├── vehicle.py            # Kinematic vehicle dynamics
│   ├── traffic_behaviors.py  # Constant / IDM / MOBIL behaviours
│   ├── traffic_manager.py    # NPC spawn, despawn, stepping
│   ├── observation.py        # Fixed-size observation builder
│   └── simulator.py          # Gym-like step/reset interface
├── agents/
│   ├── base.py               # Abstract Agent interface
│   ├── expert.py             # Rule-based expert driver
│   ├── bc_agent.py           # Behavior cloning agent
│   ├── planner_true.py       # MPC with true simulator
│   └── planner_learned.py    # MPC with learned world model
├── models/
│   ├── policy_net.py         # MLP policy network
│   └── world_model.py        # Attention-based world model
├── training/
│   ├── data_collector.py     # Trajectory collection
│   ├── bc_trainer.py         # Supervised BC training
│   ├── world_model_trainer.py# World model training
│   ├── rl_trainer.py         # REINFORCE trainer
│   └── evaluator.py          # Multi-agent evaluation & comparison
├── utils/
│   ├── types.py              # Core dataclasses (Action, VehicleState, etc.)
│   └── config.py             # All hyperparameters
├── tests/
│   ├── test_types_config.py
│   ├── test_road_vehicle.py
│   ├── test_traffic_behaviors.py
│   ├── test_traffic_manager.py
│   ├── test_observation.py
│   ├── test_simulator.py
│   ├── test_expert.py
│   ├── test_models.py
│   ├── test_training.py
│   └── test_planners_eval.py
├── run_comparison.py         # Full pipeline: train + compare all agents
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
cd selfdrive
pytest tests/ -v
```

Run individual test files:
```bash
pytest tests/test_traffic_behaviors.py -v
pytest tests/test_simulator.py -v
```

## Running the Full Comparison

```bash
python run_comparison.py
```

Options:
```bash
python run_comparison.py \
  --num-lanes 3 \
  --num-npcs 8 \
  --collect-episodes 50 \
  --bc-epochs 80 \
  --wm-epochs 80 \
  --rl-episodes 300 \
  --eval-episodes 30
```

## Key Design Decisions

### Modularity
Every module has a clean interface and its own test file. Adding a new agent means:
1. Subclass `Agent` in `agents/`
2. Add it to the comparison dict in `run_comparison.py`
3. Write tests in `tests/`

### Extensibility
- **More lanes**: Change `num_lanes` in config — all modules handle it.
- **Continuous actions**: Modify `Action` class and `VehicleDynamics` — agents and models adapt.
- **Better RL**: Swap `rl_trainer.py` internals (e.g., PPO) — interface stays the same.
- **Richer world model**: Add recurrence to `world_model.py` — planner interface unchanged.

### Action Space
9 discrete actions: 3 longitudinal (brake/keep/accelerate) × 3 lateral (left/keep/right).

### Observation Space
Fixed-size vector: `[ego_speed, ego_lane, ego_x, neighbor_1..K × (rel_x, rel_lane, rel_speed, exists)]`

### Attention World Model
Each vehicle is a token. Self-attention captures inter-vehicle interactions (e.g., braking cascade). Predicts state deltas (residual learning). Handles variable neighbor counts via attention masking.
