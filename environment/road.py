"""Road geometry: lanes, boundaries, and position queries.

This module models a straight, multi-lane road segment.  The road is
one-dimensional in the longitudinal (x) direction and discretised into
parallel lanes in the lateral direction.  Each lane is identified by a
zero-based integer index (lane 0 is the leftmost lane).

The ``Road`` object is intentionally stateless beyond its configuration;
it simply provides geometric helpers used by the simulator, traffic
manager, and renderer.
"""
from utils.config import RoadConfig


class Road:
    """Multi-lane road segment with configurable geometry.

    The road extends from ``x = 0`` to ``x = road_length`` in the
    longitudinal direction and contains ``num_lanes`` lanes, each of width
    ``lane_width`` meters.

    Attributes:
        config: A ``RoadConfig`` holding lane count, lane width, road length,
                and speed limit.
    """

    def __init__(self, config: RoadConfig = None):
        """Initialise the road.

        Args:
            config: Road geometry configuration.  Defaults to
                    ``RoadConfig()`` when *None*.
        """
        self.config = config or RoadConfig()

    @property
    def num_lanes(self) -> int:
        """Total number of parallel lanes on the road."""
        return self.config.num_lanes

    @property
    def lane_width(self) -> float:
        """Width of a single lane in meters."""
        return self.config.lane_width

    @property
    def road_length(self) -> float:
        """Total longitudinal extent of the road in meters."""
        return self.config.road_length

    @property
    def total_width(self) -> float:
        """Total lateral width of the road (all lanes combined), in meters."""
        return self.num_lanes * self.lane_width

    def lane_center_y(self, lane: int) -> float:
        """Return the lateral (Y) coordinate of a lane's centre line.

        Lane 0 spans from y=0 to y=lane_width, so its centre is at
        0.5 * lane_width.  In general: centre = (lane + 0.5) * lane_width.

        Args:
            lane: Zero-based lane index.

        Returns:
            Y-coordinate of the lane centre in meters.
        """
        return (lane + 0.5) * self.lane_width

    def is_valid_lane(self, lane: int) -> bool:
        """Check whether a lane index falls within the road boundaries.

        Args:
            lane: Integer lane index to test.

        Returns:
            *True* if ``0 <= lane < num_lanes``.
        """
        return 0 <= lane < self.num_lanes

    def is_position_on_road(self, x: float) -> bool:
        """Check whether a longitudinal position lies on the road segment.

        Args:
            x: Longitudinal position in meters.

        Returns:
            *True* if ``0 <= x <= road_length``.
        """
        return 0.0 <= x <= self.road_length

    def clamp_lane(self, lane: int) -> int:
        """Clamp a lane index to the valid range ``[0, num_lanes - 1]``.

        Args:
            lane: Possibly out-of-bounds lane index.

        Returns:
            The nearest valid lane index.
        """
        return max(0, min(lane, self.num_lanes - 1))
