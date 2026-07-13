"""Tests for ecoscope_workflows_ext_ste.tasks.spatial_operations._homerange.

`calculate_elliptical_time_density_grouped` is registered via
`wt_registry.register()`, which is a no-op at call time, so it behaves as a
plain Python function here. It groups an input trajectory GeoDataFrame by
`groupby_cols` and computes an elliptical time-density home range (via
`ecoscope.platform.tasks.analysis._time_density.calculate_elliptical_time_density`)
for each group, then concatenates the per-group results with the group key
columns attached.

The real fixture data (`sample_trajs.gpkg` / `sample_season_traj.gpkg`) are
already trajectory *segment* GeoDataFrames -- they carry the `segment_start`,
`speed_kmhr`, and LineString `geometry` columns that the underlying time
density computation requires -- so they are used directly rather than being
converted from raw relocation points.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from ecoscope_workflows_ext_ste.tasks.spatial_operations._homerange import (
    calculate_elliptical_time_density_grouped,
)

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def trajectory_gdf():
    """Real trajectory-segment fixture; a single subject, single group value."""
    return gpd.read_file(TEST_DATA_DIR / "sample_trajs.gpkg")


@pytest.fixture
def season_trajectory_gdf():
    """Real trajectory-segment fixture with an added `season` column."""
    return gpd.read_file(TEST_DATA_DIR / "sample_season_traj.gpkg")


@pytest.fixture
def two_group_trajectory_gdf(trajectory_gdf):
    """Split the real trajectory fixture into two synthetic groups by index
    parity, so grouping logic can be exercised with more than one group."""
    gdf = trajectory_gdf.copy()
    gdf["synthetic_group"] = np.where(gdf.index % 2 == 0, "A", "B")
    return gdf


class TestHappyPath:
    def test_single_group_default_args(self, trajectory_gdf):
        result = calculate_elliptical_time_density_grouped(trajectory_gdf, groupby_cols=["groupby_col"])

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {"percentile", "geometry", "area_sqkm", "groupby_col"}
        # Default percentiles=[99.9] -> exactly one output row for one group.
        assert len(result) == 1
        assert result["percentile"].iloc[0] == pytest.approx(99.9)
        assert result["groupby_col"].nunique() == 1
        assert (result["groupby_col"] == trajectory_gdf["groupby_col"].iloc[0]).all()

        geom = result.geometry.iloc[0]
        assert geom is not None
        assert geom.is_valid
        assert geom.area > 0
        assert result["area_sqkm"].iloc[0] > 0
        # Output geometry CRS is the density-calculation CRS (EPSG:3857 default),
        # not necessarily the input CRS.
        assert result.crs is not None

    def test_multiple_percentiles(self, trajectory_gdf):
        result = calculate_elliptical_time_density_grouped(
            trajectory_gdf, groupby_cols=["groupby_col"], percentiles=[50.0, 90.0]
        )
        assert len(result) == 2
        assert set(result["percentile"]) == {50.0, 90.0}
        # Higher percentile home range should enclose a larger (or equal) area.
        by_pct = result.set_index("percentile")["area_sqkm"]
        assert by_pct.loc[90.0] >= by_pct.loc[50.0]

    def test_groupby_season_column(self, season_trajectory_gdf):
        result = calculate_elliptical_time_density_grouped(season_trajectory_gdf, groupby_cols=["season"])
        assert len(result) == 1
        assert result["season"].iloc[0] == "wet"

    def test_multiple_groups_each_produce_a_block(self, two_group_trajectory_gdf):
        result = calculate_elliptical_time_density_grouped(
            two_group_trajectory_gdf, groupby_cols=["synthetic_group"], percentiles=[99.9]
        )
        # Two groups x one percentile each = two rows, one per group.
        assert len(result) == 2
        assert set(result["synthetic_group"]) == {"A", "B"}
        assert result.groupby("synthetic_group").size().to_dict() == {"A": 1, "B": 1}

    def test_multi_column_groupby(self, two_group_trajectory_gdf):
        gdf = two_group_trajectory_gdf.copy()
        gdf["always_same"] = "x"
        result = calculate_elliptical_time_density_grouped(
            gdf, groupby_cols=["synthetic_group", "always_same"], percentiles=[99.9]
        )
        assert len(result) == 2
        assert set(result.columns) >= {"synthetic_group", "always_same"}
        assert (result["always_same"] == "x").all()


class TestValidation:
    def test_missing_groupby_column_raises_value_error(self, trajectory_gdf):
        with pytest.raises(ValueError, match="Missing groupby columns"):
            calculate_elliptical_time_density_grouped(trajectory_gdf, groupby_cols=["does_not_exist"])

    def test_missing_groupby_column_error_lists_available_columns(self, trajectory_gdf):
        with pytest.raises(ValueError) as exc_info:
            calculate_elliptical_time_density_grouped(trajectory_gdf, groupby_cols=["does_not_exist"])
        message = str(exc_info.value)
        assert "does_not_exist" in message
        assert "groupby_col" in message  # a real column should be listed as available

    def test_partial_missing_columns_still_raises(self, trajectory_gdf):
        with pytest.raises(ValueError, match="Missing groupby columns"):
            calculate_elliptical_time_density_grouped(trajectory_gdf, groupby_cols=["groupby_col", "nope"])


class TestNullGroupHandling:
    def test_all_null_groups_dropped_by_default_returns_empty(self, trajectory_gdf):
        gdf = trajectory_gdf.copy()
        gdf["synthetic_group"] = None
        result = calculate_elliptical_time_density_grouped(gdf, groupby_cols=["synthetic_group"])

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        # No geometry/percentile columns since nothing was computed.
        assert not isinstance(result, gpd.GeoDataFrame) or result.crs is None or len(result.columns) == 0

    def test_all_null_groups_kept_when_drop_disabled(self, trajectory_gdf):
        gdf = trajectory_gdf.copy()
        gdf["synthetic_group"] = None
        result = calculate_elliptical_time_density_grouped(
            gdf, groupby_cols=["synthetic_group"], drop_null_groups=False
        )

        # The null value becomes its own group and is still processed.
        assert len(result) == 1
        assert result["synthetic_group"].isna().all()

    def test_partial_null_rows_dropped_by_default(self, two_group_trajectory_gdf):
        gdf = two_group_trajectory_gdf.copy()
        # Null out the group for roughly a quarter of rows.
        null_mask = gdf.index % 4 == 0
        gdf.loc[null_mask, "synthetic_group"] = None

        result = calculate_elliptical_time_density_grouped(gdf, groupby_cols=["synthetic_group"], drop_null_groups=True)
        # Only the real "A"/"B" groups remain -- no NaN group present.
        assert set(result["synthetic_group"]) == {"A", "B"}


class TestCellSizeOverride:
    def test_custom_grid_cell_size_runs(self, trajectory_gdf):
        from ecoscope.platform.tasks.analysis._time_density import CustomGridCellSize

        result = calculate_elliptical_time_density_grouped(
            trajectory_gdf,
            groupby_cols=["groupby_col"],
            cell_size=CustomGridCellSize(grid_cell_size=300),
        )
        assert len(result) == 1
        assert result["area_sqkm"].iloc[0] > 0
