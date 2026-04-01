# Self-Driving Simulator

A multi-lane highway driving simulator where agents learn to drive by imitating
an expert and planning with learned world models. Python, NumPy, PyTorch.

## Architecture

**Environment** — Gym-like simulator with `step(action) -> (obs, reward, done, info)`.
Multi-lane road with configurable geometry. Ego vehicle uses discrete 3x3 action
space (accelerate/keep/decelerate x left/keep/right = 9 actions). First-order
kinematics (Euler integration), instant lane changes. AABB collision detection.

**NPC Traffic** — Spawned around ego, despawned when far. Each NPC assigned one of:
constant-speed, IDM (Treiber 2000), or IDM+MOBIL (Kesting 2007) with per-vehicle
personality variance. Traffic manager maintains target population via respawning.

**Observations** — Fixed-size ego-centric vector: 4 ego features (speed, lane, x,
accel) + K nearest neighbors x 5 features each (rel_x, rel_lane, rel_speed,
rel_accel, exists). Zero-padded with exists flags for missing slots.

**Reward** — Speed incentive (ego_speed / speed_limit), collision penalty (-100),
lane-change penalty (-0.5), hard-brake penalty (-1.0), unsafe-lane-change penalty
(graduated two-tier based on gap size).

## Four Agents

1. **Expert** — Rule-based hierarchical driver. Physics-aware longitudinal control
   (stopping distance, cascade braking). Continuous urgency scoring for lane changes
   with two-step lookahead. All thresholds in a dedicated ExpertConfig.

2. **Behavior Cloning (BC)** — MLP policy trained via cross-entropy on expert
   demonstrations with inverse-frequency class weighting. Observations normalized
   (zero-mean, unit-variance). Rule-based safety override vetoes unsafe actions
   using raw (unnormalized) observations for physical validity.

3. **Planner (True Model)** — MPC via random shooting using the real simulator.
   Structured candidates: 9 hand-crafted maneuvers + single-action repeats +
   random fill. Horizon 30 steps. Clone/restore simulator state per rollout.

4. **Planner (Learned World Model)** — Hybrid MPC: exact ego kinematics + learned
   NPC prediction via attention world model. Short horizon (4 steps) due to error
   compounding. Same safety override as BC.

## World Model

Transformer-based: each vehicle is a token. Ego token includes one-hot action.
Type embeddings distinguish ego from NPCs. Attention masking handles variable
neighbor count. Pre-norm blocks with GELU feedforward. Predicts observation
deltas (residual). Trained with MSE on normalized observations, gradient clipping.

## Pipeline

1. Expert drives simulator → collect (obs, action, next_obs) demonstrations
2. Train BC policy on (obs → action) via cross-entropy
3. Train world model on (obs, action → next_obs) via MSE
4. Evaluate all four agents on same episodes, compare metrics

## Key Design Choices

- Fixed-size ego-centric observations (not ragged) for standard NN input
- Residual world model (predict deltas, not absolute states)
- Hybrid planner (exact ego physics + learned NPC) avoids ego error compounding
- Safety override on all learned agents using raw observations
- All configs in dataclasses with defaults; ExpertConfig for expert thresholds
- Dual README: English + Traditional Chinese, kept in sync
- Debug viewer: browser-based HTML replay of recorded episodes with expert comparison
