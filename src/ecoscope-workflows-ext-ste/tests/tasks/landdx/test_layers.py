"""Tests for ecoscope_workflows_ext_ste.tasks.landdx._layers.

`select_map_overlay` is registered via `wt_registry.register()`, a no-op
decorator at call time, so it is exercised as a plain Python function
throughout this module.

The LandDx branch of `select_map_overlay` downloads a real GeoPackage over
HTTP via `fetch_and_persist_file` (backed by `ecoscope.io.download_file`).
That call is mocked out everywhere below -- in favor of pointing straight at
a local GeoPackage fixture -- so nothing in this module makes a network
request.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from ecoscope_workflows_ext_custom.tasks.results._map import LayerDefinition, TextLayerStyle
from ecoscope_workflows_ext_ste.tasks.landdx import _layers as landdx_layers
from ecoscope_workflows_ext_ste.tasks.landdx._layers import (
    _LDX_COLOR_MAPPING,
    _LDX_URL,
    _build_ldx_layers,
    ERSpatialFeatureOverlayOption,
    LandDxOverlayOption,
    LocalFileOverlayOption,
    select_map_overlay,
)
from ecoscope_workflows_ext_ste.tasks.spatial_operations._spatial_features import (
    FeatureTypeQuery,
    LocalSpatialLayer,
    SpatialFeatureLayer,
)


def _square(x0: float, y0: float, size: float = 0.1) -> Polygon:
    return Polygon([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)])


def _bowtie(x0: float, y0: float) -> Polygon:
    """A self-intersecting (invalid) polygon."""
    return Polygon([(x0, y0), (x0 + 1, y0 + 1), (x0 + 1, y0), (x0, y0 + 1)])


@pytest.fixture
def synthetic_ldx_gdf() -> gpd.GeoDataFrame:
    """A small, hand-built LandDx-shaped GeoDataFrame covering the edge cases:

    - one row with a `type` not present in `_LDX_COLOR_MAPPING` (dropped
      entirely before styling/labelling),
    - one row with an empty geometry, and
    - one row with an invalid (self-intersecting) geometry

    -- the latter two should survive into the categorical fill/line layer but
    be excluded from the text-label layer.
    """
    return gpd.GeoDataFrame(
        {
            "type": [
                "National Park",
                "Community Conservancy",
                "National Reserve",
                "National Reserve",
                "National Reserve",
                "Unmapped Type",
            ],
            "name": [
                "Amboseli",
                "Mara North",
                "Maasai Mara",
                "Empty Geom",
                "Invalid Geom",
                "Some Ranch",
            ],
            "geometry": [
                _square(37.0, -2.5),
                _square(35.0, -1.5),
                _square(36.0, -1.0),
                Polygon(),
                _bowtie(10.0, 10.0),
                _square(1.0, 1.0),
            ],
        },
        crs="EPSG:4326",
    )


# --------------------------------------------------------------------------- #
# _build_ldx_layers                                                            #
# --------------------------------------------------------------------------- #


class TestBuildLdxLayers:
    def test_drops_rows_with_unmapped_type(self, synthetic_ldx_gdf, tmp_path):
        path = tmp_path / "ldx.gpkg"
        synthetic_ldx_gdf.to_file(path, driver="GPKG")

        layers = _build_ldx_layers(str(path))

        for layer in layers:
            if layer.geodataframe is not None and "name" in layer.geodataframe.columns:
                assert "Some Ranch" not in layer.geodataframe["name"].tolist()

    def test_returns_categorical_layer_plus_trailing_text_layer(self, synthetic_ldx_gdf, tmp_path):
        path = tmp_path / "ldx.gpkg"
        synthetic_ldx_gdf.to_file(path, driver="GPKG")

        layers = _build_ldx_layers(str(path))

        assert len(layers) >= 2
        assert all(isinstance(layer, LayerDefinition) for layer in layers)
        text_layer = layers[-1]
        assert text_layer.layer_type == "TextLayer"
        assert text_layer.legend is None
        assert isinstance(text_layer.layer_style, TextLayerStyle)

        non_text_layers = layers[:-1]
        assert all(layer.layer_type != "TextLayer" for layer in non_text_layers)

    def test_text_layer_excludes_empty_and_invalid_geometries(self, synthetic_ldx_gdf, tmp_path):
        path = tmp_path / "ldx.gpkg"
        synthetic_ldx_gdf.to_file(path, driver="GPKG")

        layers = _build_ldx_layers(str(path))
        text_layer = layers[-1]

        names = set(text_layer.geodataframe["name"].tolist())
        assert names == {"Amboseli", "Mara North", "Maasai Mara"}
        assert "Empty Geom" not in names
        assert "Invalid Geom" not in names

    def test_text_layer_geometry_is_centroid_points_in_4326(self, synthetic_ldx_gdf, tmp_path):
        path = tmp_path / "ldx.gpkg"
        synthetic_ldx_gdf.to_file(path, driver="GPKG")

        layers = _build_ldx_layers(str(path))
        text_gdf = layers[-1].geodataframe

        assert str(text_gdf.crs).upper() in ("EPSG:4326", "EPSG:4326 ")
        assert (text_gdf.geometry.geom_type == "Point").all()

    def test_categorical_layer_uses_land_use_legend_title(self, synthetic_ldx_gdf, tmp_path):
        path = tmp_path / "ldx.gpkg"
        synthetic_ldx_gdf.to_file(path, driver="GPKG")

        layers = _build_ldx_layers(str(path))
        # At least one non-text layer should carry the "Land Use" legend.
        legend_titles = {layer.legend.title for layer in layers[:-1] if layer.legend is not None}
        assert "Land Use" in legend_titles

    def test_all_unmapped_types_returns_empty_list(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            {"type": ["Ranch", "Farm"], "name": ["A", "B"], "geometry": [_square(0, 0), _square(1, 1)]},
            crs="EPSG:4326",
        )
        path = tmp_path / "ldx.gpkg"
        gdf.to_file(path, driver="GPKG")

        assert _build_ldx_layers(str(path)) == []

    def test_empty_input_file_returns_empty_list(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            {
                "type": pd.Series([], dtype="object"),
                "name": pd.Series([], dtype="object"),
                "geometry": gpd.GeoSeries([], dtype="geometry"),
            },
            crs="EPSG:4326",
        )
        path = tmp_path / "ldx.gpkg"
        gdf.to_file(path, driver="GPKG")

        assert _build_ldx_layers(str(path)) == []

    def test_missing_type_column_raises(self, tmp_path):
        # SUSPECTED BUG (documented, not fixed): `_build_ldx_layers` accesses
        # `gdf["type"]` unconditionally. A GeoPackage that doesn't have a
        # `type` column (e.g. a malformed/unexpected upstream LandDx export)
        # blows up with a bare pandas KeyError rather than a clear,
        # intentional validation error.
        gdf = gpd.GeoDataFrame({"name": ["A"], "geometry": [_square(0, 0)]}, crs="EPSG:4326")
        path = tmp_path / "ldx.gpkg"
        gdf.to_file(path, driver="GPKG")

        with pytest.raises(KeyError):
            _build_ldx_layers(str(path))

    def test_real_kenya_pa_fixture_end_to_end(self, kenya_pa_path):
        """`kenya_pa.gpkg` already has the type/name/geometry shape LandDx
        layers are expected to have, with `type` values that fully cover
        `_LDX_COLOR_MAPPING` -- exercise the real geometry-processing logic
        against realistic data (117 rows), not just tiny synthetic ones.
        """
        layers = _build_ldx_layers(str(kenya_pa_path))

        assert len(layers) >= 2
        text_layer = layers[-1]
        assert text_layer.layer_type == "TextLayer"
        # All 117 rows have a mapped type and valid, non-empty geometry.
        assert len(text_layer.geodataframe) == 117


class TestLdxColorMapping:
    def test_expected_land_use_types(self):
        assert set(_LDX_COLOR_MAPPING) == {
            "Community Conservancy",
            "National Reserve",
            "National Park",
        }


# --------------------------------------------------------------------------- #
# select_map_overlay -- LandDx branch (mocked download)                       #
# --------------------------------------------------------------------------- #


class TestSelectMapOverlayLandDx:
    def test_downloads_are_mocked_and_layers_match_direct_build(self, monkeypatch, kenya_pa_path):
        mock_fetch = MagicMock(return_value=str(kenya_pa_path))
        monkeypatch.setattr(landdx_layers, "fetch_and_persist_file", mock_fetch)

        result = select_map_overlay(option=LandDxOverlayOption(), output_dir="/tmp/ldx-cache")

        mock_fetch.assert_called_once_with(
            url=_LDX_URL,
            output_path="/tmp/ldx-cache",
            overwrite_existing=False,
            unzip=False,
            retries=2,
        )
        expected = _build_ldx_layers(str(kenya_pa_path))
        assert len(result) == len(expected)
        assert result[-1].layer_type == "TextLayer"

    def test_custom_ldx_url_is_passed_through_to_downloader(self, monkeypatch, kenya_pa_path):
        mock_fetch = MagicMock(return_value=str(kenya_pa_path))
        monkeypatch.setattr(landdx_layers, "fetch_and_persist_file", mock_fetch)

        select_map_overlay(option=LandDxOverlayOption(), ldx_url="https://example.com/custom.gpkg")

        assert mock_fetch.call_args.kwargs["url"] == "https://example.com/custom.gpkg"

    def test_no_network_call_is_made(self, monkeypatch, kenya_pa_path):
        """Guard against accidentally hitting the real Dropbox URL: patch
        `requests.Session` (used internally by `fetch_and_persist_file`) to
        blow up if anything tries a real HTTP call, and confirm the LandDx
        path still works end-to-end because `fetch_and_persist_file` itself
        is mocked out."""

        def _boom(*args, **kwargs):
            raise AssertionError("no network calls should be made in this test")

        monkeypatch.setattr("requests.Session.head", _boom)
        monkeypatch.setattr("requests.get", _boom)
        monkeypatch.setattr(landdx_layers, "fetch_and_persist_file", MagicMock(return_value=str(kenya_pa_path)))

        result = select_map_overlay(option=LandDxOverlayOption())
        assert len(result) >= 2


# --------------------------------------------------------------------------- #
# select_map_overlay -- EarthRanger Spatial Feature branch                    #
# --------------------------------------------------------------------------- #


class TestSelectMapOverlayEarthRanger:
    def test_missing_client_raises(self):
        option = ERSpatialFeatureOverlayOption(
            layers=[SpatialFeatureLayer(query=FeatureTypeQuery(feature_type="Conservancy"))]
        )
        with pytest.raises(ValueError, match="EarthRanger `client` is required"):
            select_map_overlay(option=option, client=None)

    def test_loads_each_layer_spec_and_strips_trailing_slash_from_server(self, monkeypatch):
        seen_server_urls = []

        def fake_load(self, client, server_url):
            seen_server_urls.append(server_url)
            return gpd.GeoDataFrame({"a": [1]}, geometry=[_square(0, 0)], crs="EPSG:4326")

        monkeypatch.setattr(SpatialFeatureLayer, "load", fake_load)
        monkeypatch.setattr(landdx_layers, "_layers_for_gdf", lambda gdf: [f"layer-with-{len(gdf)}-rows"])

        client = SimpleNamespace(server="http://testserver.example/api/v1/")
        option = ERSpatialFeatureOverlayOption(
            layers=[SpatialFeatureLayer(query=FeatureTypeQuery(feature_type="Conservancy"))]
        )

        result = select_map_overlay(option=option, client=client)

        assert seen_server_urls == ["http://testserver.example/api/v1"]
        assert result == ["layer-with-1-rows"]

    def test_empty_layer_results_are_skipped(self, monkeypatch):
        responses = [
            gpd.GeoDataFrame(),  # empty -> skipped
            gpd.GeoDataFrame({"a": [1]}, geometry=[_square(0, 0)], crs="EPSG:4326"),
        ]

        def fake_load(self, client, server_url):
            return responses.pop(0)

        monkeypatch.setattr(SpatialFeatureLayer, "load", fake_load)
        calls = []
        monkeypatch.setattr(
            landdx_layers,
            "_layers_for_gdf",
            lambda gdf: calls.append(gdf) or [f"layer-{len(calls)}"],
        )

        client = SimpleNamespace(server="http://testserver.example")
        option = ERSpatialFeatureOverlayOption(
            layers=[
                SpatialFeatureLayer(query=FeatureTypeQuery(feature_type="Conservancy")),
                SpatialFeatureLayer(query=FeatureTypeQuery(feature_type="Ranch")),
            ]
        )

        result = select_map_overlay(option=option, client=client)

        assert len(calls) == 1  # only the non-empty gdf reached _layers_for_gdf
        assert result == ["layer-1"]

    def test_client_missing_server_attribute_defaults_to_empty_string(self, monkeypatch):
        seen_server_urls = []

        def fake_load(self, client, server_url):
            seen_server_urls.append(server_url)
            return gpd.GeoDataFrame()

        monkeypatch.setattr(SpatialFeatureLayer, "load", fake_load)
        option = ERSpatialFeatureOverlayOption(
            layers=[SpatialFeatureLayer(query=FeatureTypeQuery(feature_type="Conservancy"))]
        )

        # client has no `.server` attribute at all
        select_map_overlay(option=option, client=object())

        assert seen_server_urls == [""]


# --------------------------------------------------------------------------- #
# select_map_overlay -- Local File branch                                     #
# --------------------------------------------------------------------------- #


class TestSelectMapOverlayLocalFile:
    def test_loads_real_local_geopackage(self, kenya_pa_path):
        option = LocalFileOverlayOption(files=[LocalSpatialLayer(file_path=str(kenya_pa_path))])

        result = select_map_overlay(option=option)

        assert len(result) >= 1
        assert all(isinstance(layer, LayerDefinition) for layer in result)
        total_rows = sum(len(layer.geodataframe) for layer in result if layer.geodataframe is not None)
        assert total_rows == 117

    def test_empty_files_list_returns_empty(self):
        option = LocalFileOverlayOption(files=[])
        assert select_map_overlay(option=option) == []

    def test_multiple_files_are_all_loaded(self, kenya_pa_path, data_dir):
        option = LocalFileOverlayOption(
            files=[
                LocalSpatialLayer(file_path=str(kenya_pa_path)),
                LocalSpatialLayer(file_path=str(data_dir / "AOIs.gpkg"), layer="AOIs"),
            ]
        )

        result = select_map_overlay(option=option)
        total_rows = sum(len(layer.geodataframe) for layer in result if layer.geodataframe is not None)
        # 117 (kenya_pa) + 7 (AOIs), potentially split across GeoJson/Icon layers.
        assert total_rows == 117 + 7


# --------------------------------------------------------------------------- #
# select_map_overlay -- unrecognized option                                   #
# --------------------------------------------------------------------------- #


class TestSelectMapOverlayFallback:
    def test_unrecognized_option_type_silently_returns_empty_list(self):
        # SUSPECTED BUG (documented, not fixed): `select_map_overlay` falls
        # through to `return []` for any `option` that isn't one of the three
        # known model types, rather than raising. This is unreachable through
        # the normal pydantic-validated entry point (the `MapOverlayOption`
        # Union only allows those three), but since the function is called
        # directly in these tests (and potentially from other Python code
        # bypassing schema validation), passing a bad/unexpected object
        # degrades silently to an empty layer list instead of erroring loudly.
        result = select_map_overlay(option=object())
        assert result == []
