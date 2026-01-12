from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_ext_ecoscope.tasks.preprocessing._preprocessing import TrajectorySegmentFilter


# creates custom trajsegfilter for this workflow and also to allow me to use the same filter on both api calls
@task
def custom_trajectory_segment_filter(
    min_length_meters: float = 0.001,
    max_length_meters: float = 5000,
    min_time_secs: float = 1,
    max_time_secs: float = 21600,
    min_speed_kmhr: float = 0.01,
    max_speed_kmhr: float = 9.0,
) -> TrajectorySegmentFilter:
    return TrajectorySegmentFilter(
        min_length_meters=min_length_meters,
        max_length_meters=max_length_meters,
        min_time_secs=min_time_secs,
        max_time_secs=max_time_secs,
        min_speed_kmhr=min_speed_kmhr,
        max_speed_kmhr=max_speed_kmhr,
    )
