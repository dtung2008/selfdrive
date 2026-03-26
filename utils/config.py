"""Central configuration for the self-driving simulator."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class RoadConfig:
    num_lanes: int = 2
    lane_width: float = 3.7        # meters
    road_length: float = 500.0     # meters (visible window)
    speed_limit: float = 30.0      # m/s  (~108 km/h)


@dataclass
class VehicleConfig:
    length: float = 4.5
    width: float = 2.0
    max_speed: float = 35.0        # m/s
    min_speed: float = 0.0
    max_acceleration: float = 3.0  # m/s^2
    max_deceleration: float = 5.0  # m/s^2 (positive value, applied as negative)
    lane_change_duration: float = 1.0  # seconds (for smooth transition)


@dataclass
class TrafficConfig:
    num_npcs: int = 8
    spawn_range: float = 200.0      # spawn NPCs within this range ahead/behind ego
    despawn_distance: float = 300.0  # remove NPCs farther than this from ego
    min_spawn_gap: float = 25.0      # minimum gap between spawned vehicles (meters)
    # IDM parameters
    idm_desired_speed: float = 28.0  # m/s
    idm_time_headway: float = 1.5    # seconds
    idm_min_gap: float = 2.0         # meters
    idm_accel: float = 2.0           # m/s^2
    idm_decel: float = 3.0           # m/s^2 (comfortable deceleration)
    idm_delta: float = 4.0           # acceleration exponent
    # MOBIL parameters
    mobil_politeness: float = 0.5
    mobil_threshold: float = 0.2     # m/s^2 — minimum advantage to change lane
    mobil_safe_decel: float = 4.0    # m/s^2 — max imposed deceleration on follower
    # Behaviour mix: probabilities [constant, idm_only, idm+mobil]
    behavior_mix: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])
    # Aggressive / timid parameter variance (multiplier std)
    personality_std: float = 0.15


@dataclass
class SimConfig:
    dt: float = 0.1                 # simulation timestep (seconds)
    episode_steps: int = 500        # max steps per episode
    collision_reward: float = -100.0
    speed_reward_scale: float = 1.0  # reward per m/s of ego speed
    lane_change_penalty: float = -0.5
    hard_brake_penalty: float = -1.0
    comfort_jerk_threshold: float = 5.0  # m/s^2 change that counts as hard brake


@dataclass
class ObservationConfig:
    k_neighbors: int = 6            # number of nearest neighbors in obs


@dataclass
class ModelConfig:
    # Policy network
    policy_hidden: int = 128
    policy_layers: int = 2
    # World model (attention)
    wm_embed_dim: int = 64
    wm_num_heads: int = 4
    wm_num_layers: int = 2
    wm_max_vehicles: int = 7  # ego + k_neighbors
    # Training
    learning_rate: float = 3e-4
    batch_size: int = 64
    bc_epochs: int = 200
    wm_epochs: int = 20
    rl_episodes: int = 500
    # Planner
    planner_horizon: int = 30           # for planner_true (uses real sim)
    planner_wm_horizon: int = 4         # for planner_learned (WM errors compound)
    planner_num_rollouts: int = 50
    planner_discount: float = 0.99


@dataclass
class Config:
    road: RoadConfig = field(default_factory=RoadConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    obs: ObservationConfig = field(default_factory=ObservationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


# Global default
DEFAULT_CONFIG = Config()
