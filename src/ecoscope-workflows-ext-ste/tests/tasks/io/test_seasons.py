"""Tests for ecoscope_workflows_ext_ste.tasks.io._seasons.

Both `compute_seasons_from_ndvi` and `create_seasonal_labels` are registered
via `wt_registry.register()`, which is a no-op at call time, so they behave
as plain Python functions here.

`compute_seasons_from_ndvi` calls Google Earth Engine (via the module's
`_fetch_ndvi`) to fetch NDVI values, which requires real network access and
EE credentials. These tests patch `_fetch_ndvi` (the only EE-touching call
in this module) with synthetic
NDVI data and let the rest of the pipeline -- `val_cuts` (sklearn Gaussian
mixture) and `seasonal_windows` (pure pandas) -- run for real, since neither
touches the network. The unused `client: EarthEngineClient` parameter is
never referenced in the function body (establishing the EE connection is a
side effect of constructing the client upstream), so tests simply pass
`None` for it.

`compute_seasons_from_ndvi` always starts the NDVI history at `MODIS_START`
(2000-02-24), whatever `time_range.since` is. That is deliberate: the season
cut is fitted on the long-term record.

A suspected source bug was found while writing these tests (not fixed here):

`create_seasonal_labels` silently produces an all-`None` `season` column
   when `trajectories["segment_start"/"segment_end"]` and
   `seasons_df["start"/"end"]` have different `datetime64` *unit* resolution
   (e.g. `datetime64[ms, UTC]`, as commonly produced by GeoPackage reads via
   pyogrio, vs `datetime64[ns, UTC]`, as produced by `pd.to_datetime` on a
   CSV) -- `pd.cut` raises internally, and the broad `except Exception` in
   `create_seasonal_labels` swallows it, printing a message but returning
   every row with `season=None` instead of raising or resampling. This is
   reproducible with the *real* `sample_season_traj.gpkg` +
   `seasonal_windows.csv` fixtures. See
   `test_real_fixtures_silently_fail_due_to_datetime_unit_mismatch`.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

import ecoscope_workflows_ext_ste.tasks.io._seasons as seasons_mod
from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange
from ecoscope_workflows_ext_ste.tasks.io._seasons import (
    MAX_CHUNK_DAYS,
    MODIS_START,
    _fetch_ndvi,
    compute_seasons_from_ndvi,
    create_seasonal_labels,
)


def _fake_fetch_ndvi(rng_seed=0, n=20):
    rng = np.random.default_rng(rng_seed)

    def _fake(img_coll, nir_band, red_band, aoi, start, end):
        dates = pd.date_range(start=start, end=end, periods=n, tz="UTC")
        ndvi = rng.uniform(0, 1, size=n)
        return pd.DataFrame({"img_date": dates, "NDVI": ndvi})

    return _fake


@pytest.fixture
def time_range() -> TimeRange:
    return TimeRange(
        since=datetime(2015, 1, 1, tzinfo=timezone.utc),
        until=datetime(2016, 1, 1, tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )


# ============================================================================
# compute_seasons_from_ndvi
# ============================================================================


class TestComputeSeasonsFromNdvi:
    def test_returns_dataframe_with_expected_columns(self, aois_gdf, time_range):
        with patch.object(seasons_mod, "_fetch_ndvi", side_effect=_fake_fetch_ndvi()):
            result = compute_seasons_from_ndvi(client=None, roi=aois_gdf, time_range=time_range, chunk_count=3)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["start", "end", "season"]
        assert set(result["season"].unique()) <= {"Dry", "Wet"}
        assert len(result) > 0

    def test_chunk_count_determines_number_of_ndvi_calls(self, aois_gdf, time_range):
        fake = _fake_fetch_ndvi()
        with patch.object(seasons_mod, "_fetch_ndvi", side_effect=fake) as mock_ndvi:
            compute_seasons_from_ndvi(client=None, roi=aois_gdf, time_range=time_range, chunk_count=20)

        # chunk_count produces chunk_count - 1 EE queries (see docstring).
        assert mock_ndvi.call_count == 19

    def test_chunks_never_cover_more_than_max_chunk_days(self, aois_gdf, time_range):
        """A low chunk_count (the mapbook spec passes 5) must not produce
        multi-year requests: the MODIS record is split further automatically."""
        fake = _fake_fetch_ndvi()
        with patch.object(seasons_mod, "_fetch_ndvi", side_effect=fake) as mock_ndvi:
            compute_seasons_from_ndvi(client=None, roi=aois_gdf, time_range=time_range, chunk_count=5)

        assert mock_ndvi.call_count > 4
        for call in mock_ndvi.call_args_list:
            span = pd.Timestamp(call.kwargs["end"]) - pd.Timestamp(call.kwargs["start"])
            assert span <= pd.Timedelta(days=MAX_CHUNK_DAYS)
        # Chunks are contiguous: each one starts where the previous one ended.
        starts = [c.kwargs["start"] for c in mock_ndvi.call_args_list]
        ends = [c.kwargs["end"] for c in mock_ndvi.call_args_list]
        assert starts[1:] == ends[:-1]

    def test_ndvi_is_scaled_once_over_all_chunks(self, aois_gdf, time_range):
        """Chunks come back unscaled and are min-max scaled together, so a
        chunk with a narrow NDVI range is not stretched to 0..1 on its own."""
        captured = {}
        real_val_cuts = seasons_mod.val_cuts

        def spy(vals, num_seasons):
            captured["ndvi"] = vals["NDVI"].to_numpy().copy()
            return real_val_cuts(vals, num_seasons)

        calls = iter(range(1000))

        def fake(img_coll, nir_band, red_band, aoi, start, end):
            i = next(calls)
            dates = pd.date_range(start=start, end=end, periods=11, tz="UTC")[:-1]
            # First chunk spans 0.2..0.8; all later chunks sit in a narrow 0.4..0.5 band.
            ndvi = np.linspace(0.2, 0.8, 10) if i == 0 else np.linspace(0.4, 0.5, 10)
            return pd.DataFrame({"img_date": dates, "NDVI": ndvi})

        with (
            patch.object(seasons_mod, "_fetch_ndvi", side_effect=fake),
            patch.object(seasons_mod, "val_cuts", side_effect=spy),
        ):
            compute_seasons_from_ndvi(client=None, roi=aois_gdf, time_range=time_range, chunk_count=2)

        scaled = captured["ndvi"]
        assert scaled.min() == pytest.approx(0.0)
        assert scaled.max() == pytest.approx(1.0)
        # The narrow chunks keep their place on the shared scale: (0.4-0.2)/0.6 .. (0.5-0.2)/0.6
        assert scaled[10:].min() == pytest.approx(1 / 3)
        assert scaled[10:].max() == pytest.approx(0.5)

    def test_no_ndvi_values_raises_clear_error(self, aois_gdf, time_range):
        def empty(img_coll, nir_band, red_band, aoi, start, end):
            return pd.DataFrame({"img_date": pd.to_datetime([], utc=True), "NDVI": np.array([], dtype=float)})

        with patch.object(seasons_mod, "_fetch_ndvi", side_effect=empty):
            with pytest.raises(ValueError, match="returned no NDVI values"):
                compute_seasons_from_ndvi(client=None, roi=aois_gdf, time_range=time_range)

    def test_since_is_always_clamped_to_modis_start_even_when_later(self, aois_gdf, time_range):
        """`since` (2015-01-01, well after MODIS coverage began) is replaced
        with MODIS_START: the season cut uses the whole MODIS record."""
        fake = _fake_fetch_ndvi()
        with patch.object(seasons_mod, "_fetch_ndvi", side_effect=fake) as mock_ndvi:
            compute_seasons_from_ndvi(client=None, roi=aois_gdf, time_range=time_range, chunk_count=2)

        first_call_start = mock_ndvi.call_args_list[0].kwargs["start"]
        assert first_call_start == MODIS_START.replace(tzinfo=timezone.utc).isoformat()

    def test_predating_modis_start_is_also_clamped_forward(self, aois_gdf):
        tr = TimeRange(
            since=datetime(1990, 1, 1, tzinfo=timezone.utc),
            until=datetime(2001, 1, 1, tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        )
        fake = _fake_fetch_ndvi()
        with patch.object(seasons_mod, "_fetch_ndvi", side_effect=fake) as mock_ndvi:
            compute_seasons_from_ndvi(client=None, roi=aois_gdf, time_range=tr, chunk_count=2)

        first_call_start = mock_ndvi.call_args_list[0].kwargs["start"]
        assert first_call_start == MODIS_START.replace(tzinfo=timezone.utc).isoformat()

    def test_passes_band_and_collection_overrides_through(self, aois_gdf, time_range):
        fake = _fake_fetch_ndvi()
        with patch.object(seasons_mod, "_fetch_ndvi", side_effect=fake) as mock_ndvi:
            compute_seasons_from_ndvi(
                client=None,
                roi=aois_gdf,
                time_range=time_range,
                img_collection="CUSTOM/COLLECTION",
                nir_band="nir",
                red_band="red",
                chunk_count=2,
            )

        call = mock_ndvi.call_args_list[0]
        assert call.kwargs["img_coll"] == "CUSTOM/COLLECTION"
        assert call.kwargs["nir_band"] == "nir"
        assert call.kwargs["red_band"] == "red"

    def test_merges_multi_polygon_roi_into_single_geometry(self, aois_gdf, time_range):
        assert len(aois_gdf) > 1  # multiple named AOI polygons
        fake = _fake_fetch_ndvi()
        with patch.object(seasons_mod, "_fetch_ndvi", side_effect=fake) as mock_ndvi:
            compute_seasons_from_ndvi(client=None, roi=aois_gdf, time_range=time_range, chunk_count=2)

        aoi_arg = mock_ndvi.call_args_list[0].kwargs["aoi"]
        # union_all() collapses the GeoDataFrame down to one geometry object.
        assert not isinstance(aoi_arg, gpd.GeoDataFrame)
        assert hasattr(aoi_arg, "geom_type")


# ============================================================================
# _fetch_ndvi -- Earth Engine request, with a stand-in `ee` module
# ============================================================================


class TestFetchNdvi:
    def _fake_ee(self, rows):
        fake_ee = MagicMock()
        chain = fake_ee.ImageCollection.return_value.select.return_value.filterDate.return_value
        mapped = chain.map.return_value.filter.return_value
        mapped.reduceColumns.return_value.get.return_value.getInfo.return_value = rows
        return fake_ee, chain

    def test_builds_dataframe_from_time_and_ndvi_pairs(self):
        rows = [[1577836800000, 0.25], [1577923200000, 0.5]]  # 2020-01-01, 2020-01-02
        fake_ee, chain = self._fake_ee(rows)
        with patch.dict("sys.modules", {"ee": fake_ee}):
            df = _fetch_ndvi(box(37, -3, 38, -2), "COLL", "nir", "red", "2020-01-01", "2020-01-03")

        assert list(df.columns) == ["img_date", "NDVI"]
        assert list(df["img_date"]) == list(pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True))
        assert list(df["NDVI"]) == [0.25, 0.5]
        fake_ee.ImageCollection.assert_called_once_with("COLL")
        fake_ee.ImageCollection.return_value.select.assert_called_once_with(["nir", "red"])
        fake_ee.ImageCollection.return_value.select.return_value.filterDate.assert_called_once_with(
            "2020-01-01", "2020-01-03"
        )
        # Images are reduced one by one; they are never stacked into one image.
        chain.toBands.assert_not_called()

    def test_each_image_is_reduced_on_its_own_at_modis_scale(self):
        fake_ee, chain = self._fake_ee([])
        with patch.dict("sys.modules", {"ee": fake_ee}):
            _fetch_ndvi(box(37, -3, 38, -2), "COLL", "nir", "red", "2020-01-01", "2020-01-03")
            per_image = chain.map.call_args.args[0]
            img = MagicMock()
            per_image(img)

        img.normalizedDifference.assert_called_once_with(["nir", "red"])
        kwargs = img.normalizedDifference.return_value.rename.return_value.reduceRegion.call_args.kwargs
        assert kwargs["scale"] == 500
        assert kwargs["bestEffort"] is True

    def test_no_images_gives_empty_frame(self):
        fake_ee, _ = self._fake_ee([])
        with patch.dict("sys.modules", {"ee": fake_ee}):
            df = _fetch_ndvi(box(37, -3, 38, -2), "COLL", "nir", "red", "2020-01-01", "2020-01-03")

        assert df.empty
        assert list(df.columns) == ["img_date", "NDVI"]


# ============================================================================
# create_seasonal_labels -- happy path
# ============================================================================


@pytest.fixture
def matching_dtype_trajectories() -> gpd.GeoDataFrame:
    """Synthetic trajectory segments with ns-precision datetimes, so their
    dtype matches `matching_dtype_seasons` exactly (see module docstring
    re: the ms-vs-ns datetime unit mismatch bug)."""
    return gpd.GeoDataFrame(
        {
            "segment_start": pd.to_datetime(["2020-01-05", "2020-02-10", "2020-03-15", "2020-06-01"], utc=True),
            "segment_end": pd.to_datetime(["2020-01-06", "2020-02-11", "2020-03-16", "2020-06-02"], utc=True),
            "geometry": [Point(0, 0)] * 4,
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def matching_dtype_seasons() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"], utc=True),
            "end": pd.to_datetime(["2020-02-01", "2020-03-01", "2020-04-01"], utc=True),
            "season": ["Dry", "Wet", "Dry"],
        }
    )


class TestCreateSeasonalLabelsHappyPath:
    def test_assigns_season_per_segment(self, matching_dtype_trajectories, matching_dtype_seasons):
        result = create_seasonal_labels(matching_dtype_trajectories, matching_dtype_seasons)

        assert list(result["season"]) == ["Dry", "Wet", "Dry"]

    def test_drops_segments_outside_all_windows(self, matching_dtype_trajectories, matching_dtype_seasons):
        # The 4th row (2020-06-01) is after the last window ends (2020-04-01).
        result = create_seasonal_labels(matching_dtype_trajectories, matching_dtype_seasons)

        assert len(result) == 3
        assert pd.Timestamp("2020-06-01", tz="UTC") not in result["segment_start"].values

    def test_does_not_mutate_the_season_windows_input(self, matching_dtype_trajectories, matching_dtype_seasons):
        original = matching_dtype_seasons.copy(deep=True)
        create_seasonal_labels(matching_dtype_trajectories, matching_dtype_seasons)
        pd.testing.assert_frame_equal(matching_dtype_seasons, original)

    def test_tz_naive_inputs_are_supported(self):
        traj = gpd.GeoDataFrame(
            {
                "segment_start": pd.to_datetime(["2020-01-05", "2020-02-10"]),
                "segment_end": pd.to_datetime(["2020-01-06", "2020-02-11"]),
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs="EPSG:4326",
        )
        seasons = pd.DataFrame(
            {
                "start": pd.to_datetime(["2020-01-01", "2020-02-01"]),
                "end": pd.to_datetime(["2020-02-01", "2020-03-01"]),
                "season": ["Dry", "Wet"],
            }
        )

        result = create_seasonal_labels(traj, seasons)

        assert list(result["season"]) == ["Dry", "Wet"]


# ============================================================================
# create_seasonal_labels -- graceful degradation (caught internally, no raise)
# ============================================================================


class TestCreateSeasonalLabelsGracefulErrors:
    """`create_seasonal_labels` wraps its entire body in a broad
    `try/except Exception` and, on failure, prints a message and returns
    `trajectories` with `season` set to `None` rather than raising -- so
    none of these cases actually raise out of the function."""

    def test_empty_trajectories_returns_with_none_season(self, matching_dtype_seasons):
        empty_traj = gpd.GeoDataFrame(columns=["segment_start", "segment_end", "geometry"])

        result = create_seasonal_labels(empty_traj, matching_dtype_seasons)

        assert result is not None
        assert "season" in result.columns
        assert len(result) == 0

    def test_empty_seasons_df_returns_with_none_season(self, matching_dtype_trajectories):
        empty_seasons = pd.DataFrame(columns=["start", "end", "season"])

        result = create_seasonal_labels(matching_dtype_trajectories, empty_seasons)

        assert result["season"].isna().all()
        assert len(result) == len(matching_dtype_trajectories)

    def test_missing_trajectory_columns_returns_with_none_season(self, matching_dtype_seasons):
        traj = gpd.GeoDataFrame(
            {"segment_start": pd.to_datetime(["2020-01-05"], utc=True), "geometry": [Point(0, 0)]},
            crs="EPSG:4326",
        )

        result = create_seasonal_labels(traj, matching_dtype_seasons)

        assert result["season"].isna().all()

    def test_missing_season_columns_returns_with_none_season(self, matching_dtype_trajectories):
        seasons = pd.DataFrame({"start": pd.to_datetime(["2020-01-01"], utc=True)})

        result = create_seasonal_labels(matching_dtype_trajectories, seasons)

        assert result["season"].isna().all()

    def test_non_datetime_trajectory_columns_returns_with_none_season(self, matching_dtype_seasons):
        traj = gpd.GeoDataFrame(
            {
                "segment_start": ["not-a-date", "also-not-a-date"],
                "segment_end": ["not-a-date", "also-not-a-date"],
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            crs="EPSG:4326",
        )

        result = create_seasonal_labels(traj, matching_dtype_seasons)

        assert result["season"].isna().all()

    def test_no_overlap_between_windows_and_trajectory_keeps_all_rows(self, matching_dtype_trajectories):
        far_future_seasons = pd.DataFrame(
            {
                "start": pd.to_datetime(["2200-01-01"], utc=True),
                "end": pd.to_datetime(["2200-02-01"], utc=True),
                "season": ["Wet"],
            }
        )

        result = create_seasonal_labels(matching_dtype_trajectories, far_future_seasons)

        assert result["season"].isna().all()
        # Unlike the "some rows unassigned" happy-path case, when *none* of
        # the windows overlap the trajectory timeframe at all, rows are kept
        # (not dropped) -- this branch returns early before the dropna call.
        assert len(result) == len(matching_dtype_trajectories)

    def test_none_trajectories_raises_uncaught_type_error(self, matching_dtype_seasons):
        """Suspected bug: the docstring says this function "Returns None if
        an error occurs", but passing `trajectories=None` crashes with an
        uncaught `TypeError` instead -- the `except` branch itself does
        `trajectories["season"] = None`, which fails on `None`."""
        with pytest.raises(TypeError):
            create_seasonal_labels(None, matching_dtype_seasons)


# ============================================================================
# create_seasonal_labels -- real fixture data
# ============================================================================


class TestCreateSeasonalLabelsWithRealFixtures:
    def test_real_fixtures_silently_fail_due_to_datetime_unit_mismatch(
        self, sample_season_traj_gdf, seasonal_windows_df
    ):
        """`sample_season_traj.gpkg` (read via pyogrio) has ms-precision
        datetimes; `seasonal_windows.csv` (parsed via `pd.to_datetime` in the
        `seasonal_windows_df` fixture) has ns-precision datetimes. Feeding
        both real fixtures through `create_seasonal_labels` together
        reproduces the swallowed-exception bug described in the module
        docstring: every row ends up with `season=None` even though the
        date ranges genuinely overlap."""
        traj = sample_season_traj_gdf.drop(columns=["season"])
        assert traj["segment_start"].dtype != seasonal_windows_df["start"].dtype

        result = create_seasonal_labels(traj, seasonal_windows_df)

        assert result["season"].isna().all()
        assert len(result) == len(traj)

    def test_real_fixtures_work_once_datetime_units_are_aligned(self, sample_season_traj_gdf, seasonal_windows_df):
        """Aligning the two frames' datetime64 unit (both to `ns`) avoids
        the bug above and lets real season assignment happen."""
        traj = sample_season_traj_gdf.drop(columns=["season"]).copy()
        traj["segment_start"] = traj["segment_start"].astype("datetime64[ns, UTC]")
        traj["segment_end"] = traj["segment_end"].astype("datetime64[ns, UTC]")

        result = create_seasonal_labels(traj, seasonal_windows_df)

        assert result["season"].notna().any()
        assert set(result["season"].dropna().unique()) <= {"wet", "dry"}
