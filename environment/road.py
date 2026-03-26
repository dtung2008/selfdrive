"""Road geometry: lanes, boundaries, position queries."""
from utils.config import RoadConfig


class Road:
    """Multi-lane road segment."""

    def __init__(self, config: RoadConfig = None):
        self.config = config or RoadConfig()

    @property
    def num_lanes(self) -> int:
        return self.config.num_lanes

    @property
    def lane_width(self) -> float:
        return self.config.lane_width

    @property
    def road_length(self) -> float:
        return self.config.road_length

    @property
    def total_width(self) -> float:
        return self.num_lanes * self.lane_width

    def lane_center_y(self, lane: int) -> float:
        """Y-coordinate of the center of the given lane."""
        return (lane + 0.5) * self.lane_width

    def is_valid_lane(self, lane: int) -> bool:
        """Check if lane index is within road bounds."""
        return 0 <= lane < self.num_lanes

    def is_position_on_road(self, x: float) -> bool:
        """Check if longitudinal position is on the road segment."""
        return 0.0 <= x <= self.road_length

    def clamp_lane(self, lane: int) -> int:
        """Clamp lane index to valid range."""
        return max(0, min(lane, self.num_lanes - 1))
