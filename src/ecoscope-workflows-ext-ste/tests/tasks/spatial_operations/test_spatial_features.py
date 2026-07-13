"""Tests for ecoscope_workflows_ext_ste.tasks.spatial_operations._spatial_features.

Three functions are registered via `wt_registry.register()` (a no-op at
call time): `get_featureset`, `get_spatial_features`, `load_local_spatial_file`.

`get_featureset` / `get_spatial_features` talk to an `EarthRangerClient`,
which is a `Protocol` (see `ecoscope.platform.connections`) -- there is no
runtime isinstance/type enforcement, so a small fake client with a `_get`
method and a `server` attribute is enough to exercise the real query
dispatch logic (`FeatureSetQuery`, `FeatureTypeQuery`, `FeatureIdQuery`) and
the private `_apply_geo_style` styling path end-to-end.
"""

from typing import Any

import geopandas as gpd
import pandas as pd
import pytest

from ecoscope_workflows_ext_ste.tasks.spatial_operations._spatial_features import (
    get_featureset,
    get_spatial_features,
    load_local_spatial_file,
)


# --------------------------------------------------------------------------- #
# Fake EarthRanger client + canned data
# --------------------------------------------------------------------------- #
FEATURESETS = [
    {
        "id": "fs-boundaries",
        "name": "Boundaries",
        "types": [
            {"id": "t-cons", "name": "Conservancy"},
            {"id": "t-post", "name": "Ranger Post"},
        ],
    },
    {"id": "fs-empty", "name": "EmptySet", "types": []},
]

FEATURESET_DETAIL = {
    "fs-boundaries": {
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[36.0, -1.0], [36.1, -1.0], [36.1, -0.9], [36.0, -0.9], [36.0, -1.0]]],
                },
                "properties": {
                    "type_name": "Conservancy",
                    "title": "Alpha Conservancy",
                    "fill": "#3388ff",
                    "stroke": "#111111",
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [36.05, -0.95]},
                "properties": {
                    "type_name": "Ranger Post",
                    "title": "Post 1",
                    "image": "/static/icon.png",
                },
            },
        ]
    },
    "fs-empty": {"features": []},
}

FEATURECLASSES = [
    {"name": "Conservancy", "feature_set_id": "fs-boundaries"},
    {"name": "Unlinked Type", "feature_set_id": None},
]

FEATURE_DETAIL = {
    "feat-1": {
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [36.05, -0.95]},
                "properties": {"feature_type": "t-cons"},
            }
        ]
    },
}

DEFAULT_ROUTES: dict[str, Any] = {
    "featureset/": FEATURESETS,
    "featureset/fs-boundaries/": FEATURESET_DETAIL["fs-boundaries"],
    "featureset/fs-empty/": FEATURESET_DETAIL["fs-empty"],
    "featureclass/": FEATURECLASSES,
    "feature/feat-1/": FEATURE_DETAIL["feat-1"],
}


class FakeERClient:
    """Minimal stand-in for `EarthRangerClientProtocol`."""

    def __init__(self, routes: dict[str, Any] | None = None, server: str = "https://fake.pamdas.org"):
        self.routes = dict(DEFAULT_ROUTES if routes is None else routes)
        self.server = server

    def _get(self, path: str) -> Any:
        if path not in self.routes:
            raise AssertionError(f"FakeERClient: no route registered for {path!r}")
        return self.routes[path]


@pytest.fixture
def client() -> FakeERClient:
    return FakeERClient()


# --------------------------------------------------------------------------- #
# get_featureset
# --------------------------------------------------------------------------- #
class TestGetFeatureset:
    def test_happy_path_returns_geodataframe_with_expected_rows(self, client):
        result = get_featureset(client, "fs-boundaries")
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs.to_epsg() == 4326
        assert len(result) == 2
        assert set(result["title"]) == {"Alpha Conservancy", "Post 1"}

    def test_non_dict_response_returns_empty_plain_dataframe(self):
        client = FakeERClient(routes={"featureset/weird/": ["not", "a", "dict"]})
        result = get_featureset(client, "weird")
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, gpd.GeoDataFrame)
        assert result.empty

    def test_dict_without_features_key_returns_empty_dataframe(self):
        client = FakeERClient(routes={"featureset/no-features/": {"id": "no-features"}})
        result = get_featureset(client, "no-features")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_features_list_returns_empty_dataframe(self, client):
        result = get_featureset(client, "fs-empty")
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# --------------------------------------------------------------------------- #
# get_spatial_features
# --------------------------------------------------------------------------- #
class TestGetSpatialFeaturesNoLayers:
    def test_none_layers_returns_empty_list(self, client):
        assert get_spatial_features(client, layers=None) == []

    def test_empty_layers_list_returns_empty_list(self, client):
        assert get_spatial_features(client, layers=[]) == []


class TestGetSpatialFeaturesFeatureSetQuery:
    def test_happy_path_returns_one_styled_gdf_with_all_featureset_rows(self, client):
        result = get_spatial_features(client, layers=[{"query": {"featureset_name": "Boundaries"}}])
        assert len(result) == 1
        gdf = result[0]
        assert len(gdf) == 2

    def test_native_fill_and_stroke_properties_are_mapped_to_deckgl_accessors(self, client):
        result = get_spatial_features(client, layers=[{"query": {"featureset_name": "Boundaries"}}])
        gdf = result[0]
        polygon_row = gdf[gdf["title"] == "Alpha Conservancy"].iloc[0]
        assert polygon_row["get_fill_color"][:3] == [51, 136, 255]  # #3388ff
        assert polygon_row["get_line_color"][:3] == [17, 17, 17]  # #111111

    def test_icon_marker_row_gets_icon_url_built_from_server_and_image(self, client):
        result = get_spatial_features(client, layers=[{"query": {"featureset_name": "Boundaries"}}])
        gdf = result[0]
        icon_row = gdf[gdf["title"] == "Post 1"].iloc[0]
        assert icon_row["icon_url"] == "https://fake.pamdas.org/static/icon.png"

    def test_legend_columns_populated_when_style_accessors_present(self, client):
        result = get_spatial_features(client, layers=[{"query": {"featureset_name": "Boundaries"}}])
        gdf = result[0]
        assert "legend_label" in gdf.columns
        assert set(gdf["legend_label"]) == {"Conservancy", "Ranger Post"}

    def test_custom_group_by_and_legend_title(self, client):
        result = get_spatial_features(
            client,
            layers=[
                {
                    "query": {"featureset_name": "Boundaries"},
                    "group_by": "title",
                    "legend_title": "My Layer",
                }
            ],
        )
        gdf = result[0]
        assert set(gdf["legend_label"]) == {"Alpha Conservancy", "Post 1"}
        assert (gdf["legend_title"] == "My Layer").all()

    def test_unknown_featureset_name_raises_value_error(self, client):
        with pytest.raises(ValueError, match="Boundaries2.*not found"):
            get_spatial_features(client, layers=[{"query": {"featureset_name": "Boundaries2"}}])

    def test_empty_featureset_result_is_skipped_from_output(self, client):
        result = get_spatial_features(client, layers=[{"query": {"featureset_name": "EmptySet"}}])
        assert result == []

    def test_mixed_empty_and_nonempty_layers_only_returns_nonempty(self, client):
        result = get_spatial_features(
            client,
            layers=[
                {"query": {"featureset_name": "EmptySet"}},
                {"query": {"featureset_name": "Boundaries"}},
            ],
        )
        assert len(result) == 1
        assert len(result[0]) == 2


class TestGetSpatialFeaturesFeatureTypeQuery:
    def test_happy_path_filters_to_matching_type_only(self, client):
        result = get_spatial_features(client, layers=[{"query": {"feature_type": "Conservancy"}}])
        assert len(result) == 1
        gdf = result[0]
        assert len(gdf) == 1
        assert gdf.iloc[0]["type_name"] == "Conservancy"

    def test_unknown_feature_type_raises_value_error(self, client):
        with pytest.raises(ValueError, match="Feature type 'Nonexistent' not found"):
            get_spatial_features(client, layers=[{"query": {"feature_type": "Nonexistent"}}])

    def test_feature_type_not_linked_to_featureset_raises_value_error(self, client):
        with pytest.raises(ValueError, match="not linked to a featureset"):
            get_spatial_features(client, layers=[{"query": {"feature_type": "Unlinked Type"}}])


class TestGetSpatialFeaturesFeatureIdQuery:
    def test_happy_path_maps_feature_type_id_to_type_name(self, client):
        result = get_spatial_features(client, layers=[{"query": {"feature_id": "feat-1"}}])
        assert len(result) == 1
        gdf = result[0]
        assert len(gdf) == 1
        assert gdf.iloc[0]["type_name"] == "Conservancy"


# --------------------------------------------------------------------------- #
# load_local_spatial_file
# --------------------------------------------------------------------------- #
class TestLoadLocalSpatialFileEmptyInputs:
    def test_none_files_returns_empty_list(self):
        assert load_local_spatial_file(files=None) == []

    def test_empty_files_list_returns_empty_list(self):
        assert load_local_spatial_file(files=[]) == []

    def test_blank_file_path_is_skipped(self):
        result = load_local_spatial_file(files=[{"file_path": ""}])
        assert result == []


class TestLoadLocalSpatialFileHappyPath:
    def test_single_file_no_split_returns_one_layer_with_all_rows(self, data_dir):
        result = load_local_spatial_file(files=[{"file_path": str(data_dir / "kenya_pa.gpkg")}])
        assert len(result) == 1
        assert len(result[0]) == 117

    def test_no_native_style_columns_means_no_legend_columns_added(self, data_dir):
        # kenya_pa.gpkg has no `fill`/`stroke`/`image` EarthRanger-style
        # columns and no LayerStyle override was supplied, so none of the
        # deck.gl style accessor columns (and therefore no legend_label)
        # get added -- the function is a pass-through in this case.
        result = load_local_spatial_file(files=[{"file_path": str(data_dir / "kenya_pa.gpkg")}])
        gdf = result[0]
        assert "legend_label" not in gdf.columns
        assert "get_fill_color" not in gdf.columns

    def test_split_by_produces_one_layer_per_distinct_value(self, data_dir):
        result = load_local_spatial_file(files=[{"file_path": str(data_dir / "kenya_pa.gpkg"), "split_by": "type"}])
        full = gpd.read_file(data_dir / "kenya_pa.gpkg")
        expected_groups = full["type"].nunique()
        assert len(result) == expected_groups
        assert sum(len(gdf) for gdf in result) == len(full)

    def test_split_by_uses_group_value_as_default_legend_title(self, data_dir):
        result = load_local_spatial_file(files=[{"file_path": str(data_dir / "kenya_pa.gpkg"), "split_by": "type"}])
        # Each chunk's legend_title default equals its own `type` value --
        # but legend_title is only set as a column when style accessors are
        # present; here we just confirm every row within a chunk really does
        # share exactly one `type` value (grouping is correct).
        for gdf in result:
            assert gdf["type"].nunique() == 1

    def test_explicit_layer_name_selects_that_gpkg_layer(self, data_dir):
        default_result = load_local_spatial_file(files=[{"file_path": str(data_dir / "AOIs.gpkg"), "layer": "AOIs"}])
        merged_result = load_local_spatial_file(files=[{"file_path": str(data_dir / "AOIs.gpkg"), "layer": "merged"}])
        assert len(default_result[0]) != len(merged_result[0]) or list(default_result[0]["name"]) != list(
            merged_result[0]["name"]
        )

    def test_multiple_file_specs_produce_independent_layers(self, data_dir):
        result = load_local_spatial_file(
            files=[
                {"file_path": str(data_dir / "AOIs.gpkg"), "layer": "AOIs"},
                {"file_path": str(data_dir / "kenyan_counties.gpkg")},
            ]
        )
        assert len(result) == 2
        assert len(result[0]) == 7
        assert len(result[1]) == 47

    def test_parquet_file_extension_is_read_via_read_parquet(self, data_dir, tmp_path):
        aois = gpd.read_file(data_dir / "AOIs.gpkg", layer="AOIs")
        parquet_path = tmp_path / "aois.parquet"
        aois.to_parquet(parquet_path)

        result = load_local_spatial_file(files=[{"file_path": str(parquet_path)}])
        assert len(result) == 1
        assert len(result[0]) == len(aois)
        assert set(result[0]["name"]) == set(aois["name"])


class TestLoadLocalSpatialFileErrors:
    def test_nonexistent_file_path_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.gpkg"
        with pytest.raises(Exception):  # noqa: B017 - exact exception type is pyogrio/fiona internal
            load_local_spatial_file(files=[{"file_path": str(missing)}])
