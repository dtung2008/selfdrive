"""Central configuration for the self-driving simulator.

This module collects every tunable parameter into a hierarchy of dataclass
configs, culminating in a single top-level :class:`Config` object.  The
hierarchy is:

- :class:`RoadConfig` -- physical road geometry (lanes, length, speed limit).
- :class:`VehicleConfig` -- ego-vehicle dynamics constraints.
- :class:`TrafficConfig` -- NPC spawning rules, IDM car-following model
  parameters, and MOBIL lane-change model parameters.
- :class:`SimConfig` -- time-stepping and reward shaping.
- :class:`ObservationConfig` -- observation-vector sizing.
- :class:`ModelConfig` -- neural-network architecture, training hyper-
  parameters, and planning horizons.

A module-level ``DEFAULT_CONFIG`` instance is provided so that callers can
import and use sensible defaults without explicit construction.
"""
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Road geometry
# ---------------------------------------------------------------------------

@dataclass
class RoadConfig:
    """Physical layout of the road.

    The road is modelled as a straight multi-lane highway.  Lanes are
    indexed left-to-right starting at 0.

    Attributes:
        num_lanes: Total number of lanes available on the highway.
        lane_width: Width of each lane in meters (US highway standard
            ~3.7 m).
        road_length: Length of the visible simulation window in meters.
            Vehicles beyond this range relative to the ego are candidates
            for despawning.
        speed_limit: Maximum legal speed in m/s.  30 m/s is roughly
            108 km/h / 67 mph.
    """
    num_lanes: int = 2
    lane_width: float = 3.7        # meters
    road_length: float = 500.0     # meters (visible window)
    speed_limit: float = 30.0      # m/s  (~108 km/h)


# ---------------------------------------------------------------------------
# Ego-vehicle dynamics
# ---------------------------------------------------------------------------

@dataclass
class VehicleConfig:
    """Dynamic limits and physical dimensions for the ego vehicle.

    These constraints are also used as *defaults* for NPC vehicles before
    personality variance (see ``TrafficConfig.personality_std``) is applied.

    Attributes:
        length: Bumper-to-bumper length in meters.
        width: Side-to-side width in meters.
        max_speed: Absolute speed cap in m/s (regardless of acceleration).
        min_speed: Minimum speed in m/s; typically 0 (no reversing).
        max_acceleration: Maximum throttle acceleration in m/s^2.
        max_deceleration: Maximum braking magnitude in m/s^2.  Stored as a
            *positive* number; the simulation applies it as a negative
            acceleration when braking.
        lane_change_duration: Time in seconds for a smooth lateral
            transition between adjacent lanes.  During this window the
            vehicle interpolates its y-position.
    """
    length: float = 4.5
    width: float = 2.0
    max_speed: float = 35.0        # m/s
    min_speed: float = 0.0
    max_acceleration: float = 3.0  # m/s^2
    max_deceleration: float = 5.0  # m/s^2 (positive value, applied as negative)
    lane_change_duration: float = 1.0  # seconds (for smooth transition)


# ---------------------------------------------------------------------------
# NPC traffic behaviour
# ---------------------------------------------------------------------------

@dataclass
class TrafficConfig:
    """NPC spawning, car-following (IDM), and lane-change (MOBIL) parameters.

    **Intelligent Driver Model (IDM)** governs longitudinal car-following.
    Each NPC maintains a desired speed, a comfortable following time-headway,
    and acceleration/deceleration comfort bounds.  The ``idm_delta`` exponent
    controls how sharply acceleration drops as speed approaches the desired
    speed (higher values produce a more abrupt transition).

    **MOBIL (Minimizing Overall Braking Induced by Lane changes)** governs
    lane-change decisions.  An NPC will change lanes only if:
    - The acceleration advantage exceeds ``mobil_threshold``.
    - The deceleration imposed on the new-lane follower stays within
      ``mobil_safe_decel``.
    - The advantage is weighted by ``mobil_politeness`` (0 = selfish,
      1 = fully considerate of other drivers).

    **Behaviour mix** assigns each NPC one of three driving strategies at
    spawn time:
    - *constant*: drive at a fixed speed, never change lanes.
    - *idm_only*: follow the IDM model but remain in the starting lane.
    - *idm+mobil*: full IDM + MOBIL behaviour.

    **Personality variance** adds per-NPC randomness by sampling a
    multiplicative factor ~N(1, personality_std) and applying it to IDM
    parameters, producing a natural spread of aggressive/timid drivers.

    Attributes:
        num_npcs: Target number of NPC vehicles to maintain on the road.
        spawn_range: NPCs are spawned within +/- this distance (meters)
            of the ego vehicle's current x position.
        despawn_distance: NPCs farther than this from ego are removed.
        min_spawn_gap: Minimum bumper-to-bumper gap (meters) required
            between any two vehicles when spawning a new NPC.
        idm_desired_speed: Target cruising speed for IDM (m/s).
        idm_time_headway: Desired time gap to the lead vehicle (seconds).
        idm_min_gap: Minimum net distance to maintain even at standstill
            (meters).
        idm_accel: Maximum comfortable acceleration in IDM (m/s^2).
        idm_decel: Comfortable deceleration in IDM (m/s^2).  This is
            *not* the emergency brake limit; it is the deceleration the
            driver is "comfortable" applying.
        idm_delta: Free-road acceleration exponent.  Controls the shape
            of the speed-dependent acceleration curve.
        mobil_politeness: Politeness factor p in [0, 1].  Trades off own
            advantage vs. inconvenience to others.
        mobil_threshold: Minimum net acceleration advantage (m/s^2) for a
            lane change to be worthwhile.
        mobil_safe_decel: Maximum deceleration (m/s^2) that a lane change
            may impose on the new follower in the target lane.
        behavior_mix: Probability weights ``[constant, idm_only,
            idm+mobil]`` used when sampling NPC driving strategies.  Must
            sum to 1.0.
        personality_std: Standard deviation of the per-NPC multiplicative
            personality factor applied to IDM parameters.  0 means all
            NPCs behave identically.
    """
    num_npcs: int = 8
    spawn_range: float = 200.0      # spawn NPCs within this range ahead/behind ego
    despawn_distance: float = 300.0  # remove NPCs farther than this from ego
    min_spawn_gap: float = 25.0      # minimum gap between spawned vehicles (meters)

    # -- IDM (Intelligent Driver Model) parameters --
    # These govern longitudinal car-following behaviour for NPCs.
    idm_desired_speed: float = 28.0  # m/s
    idm_time_headway: float = 1.5    # seconds
    idm_min_gap: float = 2.0         # meters
    idm_accel: float = 2.0           # m/s^2
    idm_decel: float = 3.0           # m/s^2 (comfortable deceleration)
    idm_delta: float = 4.0           # acceleration exponent

    # -- MOBIL (Minimizing Overall Braking Induced by Lane changes) --
    # These govern lateral lane-change decisions for NPCs.
    mobil_politeness: float = 0.5
    mobil_threshold: float = 0.2     # m/s^2 — minimum advantage to change lane
    mobil_safe_decel: float = 4.0    # m/s^2 — max imposed deceleration on follower

    # Behaviour mix: probabilities [constant, idm_only, idm+mobil]
    # "constant" NPCs hold a fixed speed; "idm_only" follow traffic but
    # never change lanes; "idm+mobil" use the full decision model.
    behavior_mix: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])

    # Aggressive / timid parameter variance (multiplier std).
    # Each NPC's IDM params are scaled by a sample from N(1, personality_std).
    personality_std: float = 0.15


# ---------------------------------------------------------------------------
# Simulation parameters & reward shaping
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Time-stepping, episode length, and reward function coefficients.

    The reward at each step is (approximately)::

        reward = speed_reward_scale * ego_speed
               + lane_change_penalty     (if ego changed lanes this step)
               + hard_brake_penalty      (if |accel_change| > comfort_jerk_threshold)
               + collision_reward        (if a collision occurred -- terminal)

    Attributes:
        dt: Physics integration timestep in seconds.  A smaller dt gives
            more precise dynamics but requires more steps per real-time
            second.
        episode_steps: Maximum number of simulation steps before the
            episode is forcibly terminated (timeout).
        collision_reward: Large negative reward applied on the terminal
            step of a collision.  Dominates the return to strongly
            discourage crashes.
        speed_reward_scale: Per-step reward multiplier on ego speed,
            incentivising faster (but safe) driving.
        lane_change_penalty: Small negative reward applied whenever the
            ego initiates a lane change, discouraging unnecessary weaving.
        hard_brake_penalty: Negative reward applied when the magnitude of
            acceleration change between consecutive steps exceeds
            ``comfort_jerk_threshold``, penalising jerky driving.
        comfort_jerk_threshold: The acceleration-change threshold (m/s^2)
            above which a braking event is considered "hard" and incurs the
            ``hard_brake_penalty``.
    """
    dt: float = 0.1                 # simulation timestep (seconds)
    episode_steps: int = 500        # max steps per episode
    collision_reward: float = -100.0
    speed_reward_scale: float = 1.0  # reward per m/s of ego speed
    lane_change_penalty: float = -0.5
    hard_brake_penalty: float = -1.0
    comfort_jerk_threshold: float = 5.0  # m/s^2 change that counts as hard brake


# ---------------------------------------------------------------------------
# Observation sizing
# ---------------------------------------------------------------------------

@dataclass
class ObservationConfig:
    """Controls the size of the fixed-length observation vector.

    The observation includes ego state (3 floats) plus ``k_neighbors``
    neighbour slots of 4 floats each, giving a total vector length of
    ``3 + k_neighbors * 4``.  See :class:`utils.types.Observation` for the
    full layout.

    Attributes:
        k_neighbors: Number of nearest-neighbour vehicle slots to include
            in the observation.  Unused slots are zero-padded with an
            ``exists`` flag of 0.
    """
    k_neighbors: int = 6            # number of nearest neighbors in obs


# ---------------------------------------------------------------------------
# Model / training / planner hyper-parameters
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Neural-network architecture, training, and planner hyper-parameters.

    This config covers three interconnected components:

    1. **Policy network** -- a simple MLP used for behavioural cloning (BC)
       and reinforcement learning.
    2. **World model** -- a Transformer-based attention model that predicts
       the next observation given the current state and action.  Used by the
       learned-model planner.
    3. **Planner** -- a sampling-based planner that evaluates random action
       sequences over a forward horizon (either in the real sim or through
       the world model) and picks the best one.

    Attributes:
        policy_hidden: Hidden-layer width of the policy MLP.
        policy_layers: Number of hidden layers in the policy MLP.
        wm_embed_dim: Embedding dimension for the world-model Transformer.
        wm_num_heads: Number of attention heads per Transformer layer.
        wm_num_layers: Number of stacked Transformer encoder layers.
        wm_max_vehicles: Maximum number of vehicle tokens the world model
            can attend over (ego + k_neighbors).
        learning_rate: Adam learning rate shared by BC, RL, and WM training.
        batch_size: Mini-batch size for all gradient-based training loops.
        bc_epochs: Number of epochs for behavioural-cloning pre-training.
        wm_epochs: Number of epochs for world-model training.
        rl_episodes: Number of on-policy episodes to collect during RL
            fine-tuning.
        planner_horizon: Rollout depth (in sim steps) for the *true-sim*
            planner, which can afford a longer horizon because the dynamics
            are exact.
        planner_wm_horizon: Rollout depth for the *learned world-model*
            planner.  Kept short (e.g. 4) because prediction errors
            compound over longer horizons.
        planner_num_rollouts: Number of random action-sequence rollouts
            evaluated per planning step.
        planner_discount: Discount factor (gamma) applied to future rewards
            during rollout evaluation.
    """
    # -- Policy network (MLP) --
    policy_hidden: int = 128
    policy_layers: int = 2

    # -- World model (Transformer / attention) --
    wm_embed_dim: int = 64
    wm_num_heads: int = 4
    wm_num_layers: int = 2
    wm_max_vehicles: int = 7  # ego + k_neighbors

    # -- Training hyper-parameters --
    learning_rate: float = 3e-4
    batch_size: int = 64
    bc_epochs: int = 200           # behavioural-cloning epochs
    wm_epochs: int = 20            # world-model training epochs
    rl_episodes: int = 500         # on-policy RL collection episodes

    # -- Planner hyper-parameters --
    planner_horizon: int = 30           # for planner_true (uses real sim)
    planner_wm_horizon: int = 4         # for planner_learned (WM errors compound)
    planner_num_rollouts: int = 50
    planner_discount: float = 0.99


# ---------------------------------------------------------------------------
# Top-level aggregated config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs.

    Each field uses ``field(default_factory=...)`` so that instantiating
    ``Config()`` with no arguments produces a complete set of sensible
    defaults.

    Attributes:
        road: Road geometry parameters.
        vehicle: Ego-vehicle dynamics limits.
        traffic: NPC traffic and driving-model parameters.
        sim: Simulation timing and reward shaping.
        obs: Observation vector sizing.
        model: Neural-network and planner hyper-parameters.
    """
    road: RoadConfig = field(default_factory=RoadConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    obs: ObservationConfig = field(default_factory=ObservationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


# Global default -- importable as ``from utils.config import DEFAULT_CONFIG``.
DEFAULT_CONFIG = Config()
