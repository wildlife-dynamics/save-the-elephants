import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from wt_task.testing import _find_example_return_path, _load_example_return

from ecoscope_workflows_ext_ste.tasks.io._seasons import create_seasonal_labels

ENV_VAR = "WT_TASK_MOCK_IO__ECOSCOPE_WORKFLOWS_EXT_STE_TASKS_IO_COMPUTE_SEASONS_FROM_NDVI"


def _example_seasons(monkeypatch) -> pd.DataFrame:
    monkeypatch.delenv(ENV_VAR, raising=False)  # use the packaged file, not a workflow's override
    path = _find_example_return_path("ecoscope_workflows_ext_ste.tasks.io", "compute_seasons_from_ndvi")
    return _load_example_return(path)


def test_compute_seasons_from_ndvi_is_mockable_io(monkeypatch):
    """Mock-IO test runs replace compute_seasons_from_ndvi with its example return."""
    seasons = _example_seasons(monkeypatch)

    assert list(seasons.columns) == ["start", "end", "season"]
    assert seasons.index.name == "unique_season"
    assert set(seasons["season"]) == {"Dry", "Wet"}
    # Contiguous windows, so every segment in their range gets a season.
    assert (seasons["start"].iloc[1:].to_numpy() == seasons["end"].iloc[:-1].to_numpy()).all()


def test_example_seasons_label_trajectories(monkeypatch):
    """The example works with create_seasonal_labels, so mock runs get real seasons."""
    seasons = _example_seasons(monkeypatch)
    starts = pd.date_range("2011-01-01", "2011-12-31", freq="7D", tz="UTC")
    trajectories = gpd.GeoDataFrame(
        {
            "segment_start": starts,
            "segment_end": starts + pd.Timedelta(hours=3),
            "geometry": [LineString([(37.5, 0.6), (37.51, 0.61)])] * len(starts),
        },
        crs="EPSG:4326",
    )

    labelled = create_seasonal_labels(trajectories=trajectories, seasons_df=seasons)

    assert len(labelled) == len(trajectories)
    assert labelled["season"].notna().all()
    assert set(labelled["season"]) == {"Dry", "Wet"}
