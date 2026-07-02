from wt_registry import register
from ecoscope.platform.tasks.preprocessing._preprocessing import (
    TrajectorySegmentFilter,
)


@register()
def trajectory_segment_filter(
    min_length_meters: float = 0.001,
    max_length_meters: float = 5000,
    min_time_secs: float = 1,
    max_time_secs: float = 21600,
    min_speed_kmhr: float = 0.01,
    max_speed_kmhr: float = 9.0,
) -> TrajectorySegmentFilter:
    """
    Build a TrajectorySegmentFilter with the project's default thresholds.

    Defined as a task so the constructed filter can be cached by the
    workflow framework and reused across multiple downstream tasks
    (e.g., separate API calls).

    Defaults (tuned for terrestrial wildlife tracking):
        - length: 1mm to 5km per segment
        - time: 1s to 6h per segment
        - speed: 0.01 to 9 km/h

    Override any of these by passing the corresponding parameter.

    Raises:
        ValueError: If any min value exceeds the corresponding max.
    """
    if min_length_meters > max_length_meters:
        raise ValueError(f"min_length_meters ({min_length_meters}) > " f"max_length_meters ({max_length_meters}).")
    if min_time_secs > max_time_secs:
        raise ValueError(f"min_time_secs ({min_time_secs}) > max_time_secs ({max_time_secs}).")
    if min_speed_kmhr > max_speed_kmhr:
        raise ValueError(f"min_speed_kmhr ({min_speed_kmhr}) > max_speed_kmhr ({max_speed_kmhr}).")

    return TrajectorySegmentFilter(
        min_length_meters=min_length_meters,
        max_length_meters=max_length_meters,
        min_time_secs=min_time_secs,
        max_time_secs=max_time_secs,
        min_speed_kmhr=min_speed_kmhr,
        max_speed_kmhr=max_speed_kmhr,
    )
