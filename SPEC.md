# Self-Driving Simulator — Project Specification

A modular simulation framework for learning and comparing self-driving control
strategies on a multi-lane highway. Implements the full pipeline: environment
simulation, expert data collection, behavioral cloning, world model training,
and multi-agent evaluation.

**Language:** Python 3.9+
**Dependencies:** numpy >= 1.21, torch >= 2.0, pytest >= 7.0 (dev)

---

## 1. Action Space & Core Types (`utils/types.py`)

### Enums
- `LongitudinalAction(IntEnum)`: DECELERATE(-1), KEEP(0), ACCELERATE(1)
- `LateralAction(IntEnum)`: LEFT(-1), KEEP(0), RIGHT(1)

### Action (dataclass)
Combines longitudinal + lateral into a single discrete action.
- `to_index() -> int`: flat index [0,8] via `(lon+1)*3 + (lat+1)`
- `from_index(idx) -> Action`: inverse mapping
- `num_actions() -> 9`: class method, always 9

Grid layout:
```
             LEFT  KEEP  RIGHT
DECELERATE    0     1     2
KEEP          3     4     5
ACCELERATE    6     7     8
```

### VehicleState (dataclass)
Fields: `x` (meters), `lane` (int, 0=leftmost), `speed` (m/s), `accel` (m/s^2),
`length` (4.5m default), `width` (2.0m default), `vehicle_id` (0=ego, 1000+=NPCs).
Method: `y_center(lane_width) -> float`.

### Observation (dataclass)
Fields: `ego_speed`, `ego_lane`, `ego_x`, `ego_accel`, `neighbors` (np.ndarray shape (k, features_per_neighbor)).
Each neighbor row: `[rel_x, rel_lane, rel_speed, rel_accel, exists]`.
- `to_vector() -> np.ndarray`: flattens to 1-D `[ego_speed, ego_lane, ego_x, ego_accel, neighbor0..., neighbor1...]`
- `vector_size(k) -> int`: `ego_features + k * features_per_neighbor`

### Transition (dataclass)
Fields: `obs` (np.ndarray), `action_idx` (int), `reward` (float), `next_obs` (np.ndarray), `done` (bool).

### Trajectory (dataclass)
Holds a list of Transitions. Method: `add(obs, action_idx, reward, next_obs, done)`.
Property: `total_reward` (sum of rewards).

---

## 2. Configuration (`utils/config.py`)

All configs are dataclasses with sensible defaults. A top-level `Config` aggregates all sub-configs. A `DEFAULT_CONFIG` global instance is provided.

### RoadConfig
- `num_lanes`: 2, `lane_width`: 3.7m, `road_length`: 500m, `speed_limit`: 30 m/s

### VehicleConfig
- `length`: 4.5m, `width`: 2.0m
- `max_speed`: 35 m/s, `min_speed`: 0 m/s
- `max_acceleration`: 3.0 m/s^2, `max_deceleration`: 5.0 m/s^2
- `lane_change_duration`: 1.0s

### TrafficConfig
- **Spawning:** `num_npcs` (8), `spawn_range` (+/-200m), `despawn_distance` (300m), `min_spawn_gap` (25m)
- **IDM:** `idm_desired_speed` (28), `idm_time_headway` (1.5s), `idm_min_gap` (2m), `idm_accel` (2), `idm_decel` (3), `idm_delta` (4)
- **MOBIL:** `mobil_politeness` (0.5), `mobil_threshold` (0.2), `mobil_safe_decel` (4)
- **Behavior mix:** `[0.2, 0.3, 0.5]` for [constant, idm_only, idm+mobil]
- **Personality variance:** `personality_std` (0.15) — Gaussian multiplier on per-NPC IDM params

### SimConfig
- `dt`: 0.1s, `episode_steps`: 500
- **Rewards:** `collision_reward` (-100), `speed_reward_scale` (1.0), `lane_change_penalty` (-0.5), `hard_brake_penalty` (-1.0), `comfort_jerk_threshold` (5.0 m/s^2)

### ObservationConfig
- `k_neighbors`: 10, `ego_features`: 4, `features_per_neighbor`: 5

### ExpertConfig
- **Longitudinal:** `safe_distance` (20m), `desired_speed` (30), `stopping_buffer` (10m), `max_decel` (5)
- **Lane changes:** `lane_change_gap` (15m), `lane_change_cooldown` (10 steps), `min_improvement` (10m), `closing_speed_threshold` (2)
- **Urgency escalation:** `urgency_escalation_steps` (50), `max_urgency_escalation` (0.5)
- **Cascade detection:** `cascade_accel_threshold` (-2), `cascade_max_gap` (40m)
- **Gap thresholds:** `emergency_gap_ahead/behind` (10m/5m), `normal_min_gap_ahead/behind` (20m/15m), `critical_gap` (8m)

### ModelConfig
- **Policy MLP:** `policy_hidden` (128), `policy_layers` (2)
- **World Model:** `wm_embed_dim` (64), `wm_num_heads` (4), `wm_num_layers` (2), `wm_max_vehicles` (11)
- **Training:** `learning_rate` (3e-4), `batch_size` (64), `bc_epochs` (200), `wm_epochs` (20)
- **Planning:** `planner_horizon` (30), `planner_wm_horizon` (4), `planner_num_rollouts` (50), `planner_discount` (0.99)

---

## 3. Environment

### Road (`environment/road.py`)
Stateless helper. Methods: `lane_center_y(lane)`, `is_valid_lane(lane)`, `is_position_on_road(x)`, `clamp_lane(lane)`.

### VehicleDynamics (`environment/vehicle.py`)
First-order kinematics, Euler integration, discrete lane changes.
- `step(state, action, dt, num_lanes) -> VehicleState`: for ego (discrete Action)
- `step_with_acceleration(state, accel, target_lane, dt, num_lanes) -> VehicleState`: for NPCs (continuous)
- Longitudinal: `speed' = clamp(speed + accel*dt, [min, max])`, `x' = x + speed'*dt`
- Lateral: instant discrete lane change clamped to [0, num_lanes-1]

### Traffic Behaviors (`environment/traffic_behaviors.py`)

Abstract base: `TrafficBehavior.get_action(vehicle, all_vehicles, num_lanes) -> (accel, target_lane)`

Helpers: `_find_leader()`, `_find_follower()`, `_gap()` (bumper-to-bumper).

**ConstantSpeedBehavior:** P-controller to maintain fixed speed, never changes lanes.

**IDMBehavior:** Intelligent Driver Model (Treiber et al., 2000).
- `a = a_max * [1 - (v/v0)^delta - (s*/s)^2]`
- `s* = s0 + v*T + v*dv / (2*sqrt(a*b))`
- `IDMParams` dataclass with `with_personality(rng)` for per-NPC variance.

**MOBILBehavior:** Lane-change decision model (Kesting et al., 2007) wrapping IDM.
- Safety: new follower's decel < b_safe
- Incentive: `gain = (a_new - a_cur) + p * (follower_gain) > threshold`

### Traffic Manager (`environment/traffic_manager.py`)
Manages NPC lifecycle:
- `reset(ego_x, ego_speed)`: clear NPCs, spawn initial population around ego
- `step(ego_state, dt)`: compute NPC actions via behaviors, step dynamics, despawn far NPCs, respawn to maintain count
- Spawning: rejection sampling to enforce `min_spawn_gap`
- Behavior assignment: sample from `behavior_mix`, use spawn speed as IDM desired_speed

### Observation Builder (`environment/observation.py`)
- Sorts all other vehicles by distance to ego
- Fills k nearest into fixed-size array with relative features
- Padding: zeros with `exists=0.0` flag

### Simulator (`environment/simulator.py`)
Gym-like interface.
- `reset() -> obs_vector`: ego at road_length*0.3, random lane, speed=20
- `step(action) -> (obs, reward, done, info)`: ego dynamics -> NPC dynamics -> collision check -> reward -> termination check
- `clone_state() / restore_state(snapshot)`: for tree-search planners
- `get_all_vehicle_states() -> (ego, npcs)`

**Collision detection:** AABB — same lane and center-to-center distance < sum of half-lengths.

**Reward function:**
1. Collision: -100 (terminal)
2. Speed: `ego_speed / speed_limit` (capped at 1.0); above limit: linear penalty
3. Lane change: -0.5 if lateral != KEEP
4. Unsafe lane change (two-tier): hard zone (gap < hard_safe) = -100, soft zone = graduated fraction. Separate ahead/behind thresholds.
5. Hard brake: -1.0 if |accel_change| > comfort_jerk_threshold

---

## 4. Agents

### Base (`agents/base.py`)
Abstract class: `act(obs: np.ndarray) -> Action` (required), `reset()` (optional no-op).

### Expert (`agents/expert.py`)
Rule-based hierarchical driver.

**State:** step count, last lane-change step, stuck counter.

**`act(obs)`:**
1. Parse obs into ego state + neighbor dicts (rel_x, rel_lane, rel_speed, rel_accel)
2. Find leader, 2nd leader, follower in same lane
3. Longitudinal decision (physics-based):
   - Stopping distance: `closing_speed^2 / (2*max_decel) + buffer`
   - Cascade braking: if 2nd leader decelerating, increase safe distance
   - Three states: CLOSING (decel), STABLE GAP (keep), CRUISE (match desired_speed)
4. Lateral decision (continuous urgency):
   - Urgency sources: time_to_safe proximity, speed deficit, leader deceleration, tailgater pressure
   - For each adjacent lane: check safety, compute gap improvement, apply urgency scaling
   - Two-step lookahead: credit intermediate lanes for far-lane access
   - Prefer lane with more room; random tiebreak when gaps similar
   - Lane-change cooldown (10 steps between changes)

**`_lane_is_safe(neighbors, lane_offset)`:** Physics-based gap checks with urgency escalation relaxation.

### Behavior Cloning (`agents/bc_agent.py`)
Neural network policy + rule-based safety veto.

**`act(obs)`:**
1. Normalize obs (if normalizer provided)
2. Query PolicyNetwork: deterministic=argmax or stochastic=sample
3. Convert index to Action
4. Apply safety override on raw (unnormalized) obs

**Safety override:**
- Lane-change blocking: physics-based stopping distance for ahead, fixed threshold for behind
- Three-state following distance: CLOSING (enforce safe gap), STABLE (cap at KEEP), CRITICAL (force DECEL)
- Emergency mode: relaxed thresholds when same-lane gap < 10m

### Planner True Model (`agents/planner_true.py`)
MPC via random shooting with real simulator.

**9 hand-crafted maneuvers:** keep_speed, accelerate, brake, pass_left/right, escape_left/right, brake_left/right.

**`act(obs)`:**
1. Clone simulator state
2. Build candidates (3 tiers up to `planner_num_rollouts`): maneuvers, single-action repeats, random sequences
3. Rollout each: restore state, step simulator for horizon steps, accumulate discounted reward, stop on collision
4. Select best with tie-breaking: prefer lateral=KEEP among max-reward candidates
5. Restore simulator state, return first action of best sequence

### Planner Learned Model (`agents/planner_learned.py`)
Hybrid MPC: exact ego kinematics + learned NPC prediction.

**`act(obs)`:**
1. Generate candidates: single-action repeats + random sequences
2. Parse initial state from observation (absolute positions from relative + ego)
3. For each planning step:
   - Apply exact ego kinematics (accel, speed clamp, position update, lane change)
   - Construct WM input observation (relative to current ego)
   - Run world model (normalize -> forward -> denormalize)
   - Extract NPC position deltas from WM prediction
   - Compute reward via `_estimate_reward()`
   - Accumulate discounted rewards
4. Select best with KEEP preference, apply lane-change cooldown
5. Apply safety override on raw observation

**`_estimate_reward()`:**
- Speed reward: ego_speed / speed_limit
- Safety: collision penalty (-100), too-close penalty (-10), unsafe lane-change (-100)
- Separate thresholds for ahead/behind gaps

**Short horizon (4 steps)** due to WM prediction error compounding.

---

## 5. Neural Network Models

### Policy Network (`models/policy_net.py`)
MLP: `Linear(obs_dim, hidden) -> ReLU -> ... -> Linear(hidden, 9)`.
- `forward(obs) -> logits`
- `get_action_probs(obs) -> softmax(logits)`
- `sample_action(obs) -> int` (Categorical sample)
- `greedy_action(obs) -> int` (argmax)

### Attention World Model (`models/world_model.py`)
Transformer-based dynamics model: `(obs, action) -> next_obs`.

**Components:**
- `VehicleEmbedding`: Linear projection to embed_dim
- `MultiHeadSelfAttention`: Fused QKV, scaled dot-product, padding mask (True=valid, False=padding -> -inf scores)
- `TransformerBlock`: Pre-norm (LayerNorm before sublayer), GELU feedforward

**Architecture:**
- Ego token: `[ego_features | one_hot_action]` -> ego_embed + type_embed[0]
- Neighbor tokens: `[features_per_neighbor]` -> neighbor_embed + type_embed[1]
- Type embedding: `nn.Embedding(2, embed_dim)`, added (not concatenated)
- Sequence: `[ego_token, neighbor_0, ..., neighbor_k-1]` shape (batch, 1+k, embed_dim)
- Attention mask from `exists` flag
- Output heads: `ego_head(embed_dim -> ego_features)`, `neighbor_head(embed_dim -> features_per_neighbor - 1)`

**Residual prediction:** Model predicts deltas. `pred = input + delta`. Exists flag copied unchanged.

**`forward_npc_deltas(obs, action) -> (batch, k, 3)`:** Returns only NPC deltas [rel_x, rel_lane, rel_speed] for hybrid planner.

---

## 6. Training

### Data Collector (`training/data_collector.py`)
- `DataCollector(sim, agent)`: rolls out agent in sim, returns Trajectory
- `collect_episodes(n) -> List[Trajectory]`
- `trajectories_to_arrays(trajs) -> (obs, actions, rewards, next_obs, dones)`: flatten to numpy arrays

### BC Trainer (`training/bc_trainer.py`)
- `ObsNormalizer`: fit mean/std on training data, normalize/denormalize
- `BCTrainer(policy, lr, batch_size)`:
  - Computes inverse-frequency class weights for imbalanced actions
  - Loss: `CrossEntropyLoss(weight=class_weights)`
  - `train(obs, actions, epochs) -> (losses, normalizer)`
  - Fits normalizer, creates DataLoader(shuffle=True), trains with Adam

### World Model Trainer (`training/world_model_trainer.py`)
- `ObsNormalizer`: fit on concatenation of obs+next_obs (shared statistics), with `inverse_transform()`
- `WorldModelTrainer(wm, lr, batch_size)`:
  - Loss: `MSELoss` on normalized observations
  - Gradient clipping: L2 norm <= 1.0
  - `train(obs, actions, next_obs, epochs) -> losses`

### Evaluator (`training/evaluator.py`)
- `EvalMetrics` dataclass: avg_reward, std_reward, avg_speed, collision_rate, avg_episode_length, lane_changes_per_episode, completion_rate
- `Evaluator(sim, num_episodes)`:
  - `evaluate(agent) -> EvalMetrics`
  - `print_comparison(results_dict)`: pretty-print table

---

## 7. Main Scripts

### `run_comparison.py`
Full 5-stage pipeline:
1. Create expert agent
2. Collect expert demonstrations (50 episodes)
3. Train BC (200 epochs, cross-entropy with class weights)
4. Train world model (20 epochs, MSE on normalized obs)
5. Build planners + evaluate all 4 agents

CLI: `--num-lanes`, `--num-npcs`, `--seed`, `--collect-episodes`, `--bc-epochs`, `--wm-epochs`, `--eval-episodes`, `--planner-horizon`, `--planner-rollouts`

Note: planner_true must reuse its own simulator instance during evaluation (it mutates sim state during planning).

### `record_debug.py`
Records a single episode as JSON for the debug viewer. Supports all 4 agent types. Captures per-frame: ego state, NPC states, agent action, expert action, lane gaps, disagree flag, cumulative reward.

CLI: `--agent {expert,bc,planner_true,planner_wm}`, `--seed`, `--train-seed`, `--num-lanes`, `--num-npcs`, `--max-steps`, `--output`, plus training params.

### `debug_viewer.html`
Browser-based interactive episode viewer.
- Top-down canvas: ego-relative road view, color-coded ego (green/orange/red)
- Controls: play/pause, step forward/back, scrubber, speed slider
- Jump buttons: next disagree, jump to crash
- Left panel: step metrics (speed, lane, action, gaps, reward)
- Right panel: vehicle list sorted by position
- Dump functions for text export

### Other scripts
- `run_simple_debug.py`: 1-lane comparison for longitudinal control isolation
- `record_episodes.py`: basic episode recording (less detail than record_debug)
- `trace_bc_collision.py`: step-by-step BC collision diagnosis
- `test_wm_horizon.py`: WM error compounding across horizons
- `verify_wm_rollout.py`: hybrid WM rollout accuracy vs true simulator
- `debug_planner_decision.py`: planner reward breakdown at slow steps
- `scan_failures.py`: scan seed ranges for agent failure episodes

---

## 8. Test Suite (`tests/`)

103 tests across 10 files:
- `test_types_config.py`: Action encoding round-trip, VehicleState, Observation, Config defaults
- `test_road_vehicle.py`: Road geometry, vehicle dynamics step functions
- `test_traffic_behaviors.py`: IDM acceleration, MOBIL lane-change decisions, constant-speed
- `test_traffic_manager.py`: NPC spawn/despawn, gap enforcement, multi-lane
- `test_observation.py`: Observation shape, distance sorting, padding, truncation at k
- `test_simulator.py`: Step/reset, collision detection, reward computation, clone/restore
- `test_expert.py`: Expert decisions on crafted observations
- `test_models.py`: PolicyNetwork + AttentionWorldModel output shapes, gradient flow
- `test_training.py`: DataCollector, BCTrainer loss decrease, WorldModelTrainer loss decrease
- `test_planners_eval.py`: PlannerTrue, PlannerLearned, Evaluator metrics

---

## 9. Key Design Decisions

1. **Fixed-size observations:** Zero-padded with exists flags, not ragged arrays. Enables standard MLP/transformer input.
2. **Ego-centric frame:** All neighbor features relative to ego. Translation-invariant.
3. **Residual world model:** Predicts deltas (next - current). Easier to learn when consecutive states are similar.
4. **Hybrid planner:** Exact ego kinematics + learned NPC prediction. Avoids compounding ego errors.
5. **Safety override layer:** Rule-based veto on learned agent actions. Operates on raw observations for physical validity.
6. **Separate behavior assignment per NPC:** Each NPC gets its own IDM params via personality variance.
7. **Pre-norm transformer:** LayerNorm before attention/FFN sublayers for training stability.
8. **Inverse-frequency class weighting in BC:** Rare actions (hard brake, lane change) get proportional learning signal.
9. **Structured planning candidates:** Hand-crafted maneuvers + single-action repeats + random fill. Reduces variance vs pure random shooting.
10. **Clone/restore for tree search:** Simulator supports full state snapshots for planner rollouts.

---

## 10. Dual README

The project maintains both `README.md` (English) and `讀我.md` (Traditional Chinese) with identical content structure. Both must be kept in sync when updating documentation.

---

## 11. Project Structure

```
selfdrive/
  utils/
    __init__.py
    types.py              # Action, VehicleState, Observation, Transition, Trajectory
    config.py             # All config dataclasses + DEFAULT_CONFIG
  environment/
    __init__.py
    road.py               # Road geometry
    vehicle.py            # VehicleDynamics
    traffic_behaviors.py  # IDM, MOBIL, ConstantSpeed
    traffic_manager.py    # NPC lifecycle management
    observation.py        # ObservationBuilder
    simulator.py          # Main Simulator (step/reset/reward/collision)
  agents/
    __init__.py
    base.py               # Abstract Agent
    expert.py             # Rule-based expert driver
    bc_agent.py           # Behavior cloning agent + safety override
    planner_true.py       # MPC with true simulator
    planner_learned.py    # MPC with learned world model
  models/
    __init__.py
    policy_net.py         # MLP policy network
    world_model.py        # Attention-based world model
  training/
    __init__.py
    data_collector.py     # Expert data collection
    bc_trainer.py         # BC training + ObsNormalizer
    world_model_trainer.py # WM training + ObsNormalizer
    evaluator.py          # Multi-agent evaluation
  tests/
    __init__.py
    test_types_config.py
    test_road_vehicle.py
    test_traffic_behaviors.py
    test_traffic_manager.py
    test_observation.py
    test_simulator.py
    test_expert.py
    test_models.py
    test_training.py
    test_planners_eval.py
  run_comparison.py       # Full pipeline
  record_debug.py         # Debug episode recorder
  debug_viewer.html       # Browser-based episode viewer
  README.md               # English documentation
  讀我.md                  # Traditional Chinese documentation
```
