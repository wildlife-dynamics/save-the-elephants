"""Tests for ecoscope_workflows_ext_ste.tasks.landdx._layers.

`select_map_overlay` is registered via `wt_registry.register()`, a no-op
decorator at call time, so it is exercised as a plain Python function
throughout this module.

The LandDx branch of `select_map_overlay` reads the GeoPackage shipped in
`ecoscope_workflows_ext_ste/data` (or an explicit `ldx_path`), so nothing in
this module makes a network request.
"""

from unittest.mock import MagicMock

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from ecoscope.platform.tasks.results._pydeck import (
    TextLayerStyle,
)
from ecoscope.platform.tasks.results._pydeck import (
    PydeckLayerDefinition as LayerDefinition,
)
from ecoscope_workflows_ext_ste.tasks.landdx import _layers as landdx_layers
from ecoscope_workflows_ext_ste.tasks.landdx._layers import (
    _LDX_COLOR_MAPPING,
    _build_ldx_layers,
    ERSpatialFeatureGroupOverlayOption,
    LandDxOverlayOption,
    MapOverlayOption,
    select_map_overlay,
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
    def test_drops_rows_with_unmapped_type(self, synthetic_ldx_gdf):
        layers = _build_ldx_layers(synthetic_ldx_gdf)

        for layer in layers:
            if layer.geodataframe is not None and "name" in layer.geodataframe.columns:
                assert "Some Ranch" not in layer.geodataframe["name"].tolist()

    def test_returns_categorical_layer_plus_trailing_text_layer(self, synthetic_ldx_gdf):
        layers = _build_ldx_layers(synthetic_ldx_gdf)

        assert len(layers) >= 2
        assert all(isinstance(layer, LayerDefinition) for layer in layers)
        text_layer = layers[-1]
        assert text_layer.layer_type == "TextLayer"
        assert text_layer.legend is None
        assert isinstance(text_layer.layer_style, TextLayerStyle)

        non_text_layers = layers[:-1]
        assert all(layer.layer_type != "TextLayer" for layer in non_text_layers)

    def test_text_layer_excludes_empty_and_invalid_geometries(self, synthetic_ldx_gdf):
        layers = _build_ldx_layers(synthetic_ldx_gdf)
        text_layer = layers[-1]

        names = set(text_layer.geodataframe["name"].tolist())
        assert names == {"Amboseli", "Mara North", "Maasai Mara"}
        assert "Empty Geom" not in names
        assert "Invalid Geom" not in names

    def test_text_layer_geometry_is_centroid_points_in_4326(self, synthetic_ldx_gdf):
        layers = _build_ldx_layers(synthetic_ldx_gdf)
        text_gdf = layers[-1].geodataframe

        assert str(text_gdf.crs).upper() in ("EPSG:4326", "EPSG:4326 ")
        assert (text_gdf.geometry.geom_type == "Point").all()

    def test_categorical_layer_uses_land_use_legend_title(self, synthetic_ldx_gdf):
        layers = _build_ldx_layers(synthetic_ldx_gdf)
        # At least one non-text layer should carry the "Land Use" legend.
        legend_titles = {layer.legend.title for layer in layers[:-1] if layer.legend is not None}
        assert "Land Use" in legend_titles

    def test_all_unmapped_types_returns_empty_list(self):
        gdf = gpd.GeoDataFrame(
            {"type": ["Ranch", "Farm"], "name": ["A", "B"], "geometry": [_square(0, 0), _square(1, 1)]},
            crs="EPSG:4326",
        )

        assert _build_ldx_layers(gdf) == []

    def test_empty_input_file_returns_empty_list(self):
        gdf = gpd.GeoDataFrame(
            {
                "type": pd.Series([], dtype="object"),
                "name": pd.Series([], dtype="object"),
                "geometry": gpd.GeoSeries([], dtype="geometry"),
            },
            crs="EPSG:4326",
        )

        assert _build_ldx_layers(gdf) == []

    def test_missing_type_column_raises(self):
        # SUSPECTED BUG (documented, not fixed): `_build_ldx_layers` accesses
        # `gdf["type"]` unconditionally. A GeoPackage that doesn't have a
        # `type` column (e.g. a malformed/unexpected upstream LandDx export)
        # blows up with a bare pandas KeyError rather than a clear,
        # intentional validation error.
        gdf = gpd.GeoDataFrame({"name": ["A"], "geometry": [_square(0, 0)]}, crs="EPSG:4326")

        with pytest.raises(KeyError):
            _build_ldx_layers(gdf)

    def test_real_kenya_pa_fixture_end_to_end(self, kenya_pa_path):
        """`kenya_pa.gpkg` already has the type/name/geometry shape LandDx
        layers are expected to have, with `type` values that fully cover
        `_LDX_COLOR_MAPPING` -- exercise the real geometry-processing logic
        against realistic data (117 rows), not just tiny synthetic ones.
        """
        layers = _build_ldx_layers(gpd.read_file(kenya_pa_path))

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
# select_map_overlay -- LandDx branch                                         #
# --------------------------------------------------------------------------- #


class TestSelectMapOverlayLandDx:
    def test_defaults_to_packaged_geopackage(self, monkeypatch):
        """The LandDx file ships with ext-ste, so no network is needed."""

        def _boom(*args, **kwargs):
            raise AssertionError("no network calls should be made in this test")

        monkeypatch.setattr("requests.Session.head", _boom)
        monkeypatch.setattr("requests.get", _boom)

        result = select_map_overlay(option=LandDxOverlayOption())

        assert len(result) >= 2
        assert result[-1].layer_type == "TextLayer"

    def test_custom_ldx_path_is_used(self, kenya_pa_path):
        result = select_map_overlay(option=LandDxOverlayOption(), ldx_path=f"file://{kenya_pa_path}")

        expected = _build_ldx_layers(gpd.read_file(kenya_pa_path))
        assert len(result) == len(expected)
        assert len(result[-1].geodataframe) == 117

    def test_packaged_file_is_loaded_when_no_path_given(self, monkeypatch, kenya_pa_path):
        mock_load = MagicMock(return_value=str(kenya_pa_path))
        monkeypatch.setattr(landdx_layers, "load_mapbook_file", mock_load)

        select_map_overlay(option=LandDxOverlayOption())

        mock_load.assert_called_once_with("mapbook_ldx_layers.gpkg")


# --------------------------------------------------------------------------- #
# select_map_overlay -- EarthRanger Spatial Feature Groups branch             #
# --------------------------------------------------------------------------- #


def _er_group_gdf(n: int = 2) -> gpd.GeoDataFrame:
    """Features as EarthRanger returns them: native fill/stroke properties."""
    return gpd.GeoDataFrame(
        {
            "title": [f"Feature {i}" for i in range(n)],
            "fill": ["#ff0000"] * n,
            "fill-opacity": [0.5] * n,
            "stroke": ["#000000"] * n,
            "stroke-width": [2] * n,
        },
        geometry=[_square(i, 0) for i in range(n)],
        crs="EPSG:4326",
    )


def _client(groups: dict[str, gpd.GeoDataFrame], server: str = "https://ste.pamdas.org/") -> MagicMock:
    client = MagicMock()
    client.server = server
    client.get_spatial_features_group.side_effect = lambda spatial_features_group_name, **_: groups[
        spatial_features_group_name
    ]
    return client


class TestSelectMapOverlayEarthRanger:
    def test_missing_client_raises(self):
        option = ERSpatialFeatureGroupOverlayOption(spatial_feature_groups=["Conservancies"])
        with pytest.raises(ValueError, match="EarthRanger `client` is required"):
            select_map_overlay(option=option, client=None)

    def test_requires_at_least_one_group(self):
        with pytest.raises(ValueError):
            ERSpatialFeatureGroupOverlayOption(spatial_feature_groups=[])

    def test_fetches_each_group_by_name(self):
        client = _client({"Conservancies": _er_group_gdf(), "Ranches": _er_group_gdf(1)})
        option = ERSpatialFeatureGroupOverlayOption(spatial_feature_groups=["Conservancies", "Ranches"])

        result = select_map_overlay(option=option, client=client)

        names = [c.kwargs["spatial_features_group_name"] for c in client.get_spatial_features_group.call_args_list]
        assert names == ["Conservancies", "Ranches"]
        for call in client.get_spatial_features_group.call_args_list:
            assert call.kwargs["spatial_features_group_id"] is None
            assert call.kwargs["with_group_data"] is False
        assert len(result) == 2
        assert all(layer.layer_type == "GeoJsonLayer" for layer in result)

    def test_group_name_becomes_legend_title(self):
        client = _client({"Conservancies": _er_group_gdf()})
        option = ERSpatialFeatureGroupOverlayOption(spatial_feature_groups=["Conservancies"])

        (layer,) = select_map_overlay(option=option, client=client)

        assert set(layer.geodataframe["legend_title"]) == {"Conservancies"}
        assert list(layer.geodataframe["legend_label"]) == ["Feature 0", "Feature 1"]

    def test_empty_groups_are_skipped(self):
        client = _client({"Empty": gpd.GeoDataFrame(), "Conservancies": _er_group_gdf()})
        option = ERSpatialFeatureGroupOverlayOption(spatial_feature_groups=["Empty", "Conservancies"])

        result = select_map_overlay(option=option, client=client)

        assert len(result) == 1

    def test_style_override_replaces_native_colours(self):
        client = _client({"Conservancies": _er_group_gdf()})
        option = ERSpatialFeatureGroupOverlayOption.model_validate(
            {
                "spatial_feature_groups": ["Conservancies"],
                "style": [{"polygon": [{"fill_color": ["#00ff00"]}]}],
            }
        )

        (layer,) = select_map_overlay(option=option, client=client)

        assert layer.geodataframe["get_fill_color"].iloc[0][:3] == [0, 255, 0]

    def test_union_parses_both_options(self):
        from pydantic import TypeAdapter

        adapter = TypeAdapter(MapOverlayOption)
        assert isinstance(adapter.validate_python({}), LandDxOverlayOption)
        assert isinstance(
            adapter.validate_python({"spatial_feature_groups": ["Conservancies"]}),
            ERSpatialFeatureGroupOverlayOption,
        )


# --------------------------------------------------------------------------- #
# select_map_overlay -- unrecognized option                                   #
# --------------------------------------------------------------------------- #


class TestSelectMapOverlayFallback:
    def test_unrecognized_option_type_silently_returns_empty_list(self):
        # SUSPECTED BUG (documented, not fixed): `select_map_overlay` falls
        # through to `return []` for any `option` that isn't a
        # `LandDxOverlayOption`, rather than raising. This is unreachable through
        # the normal pydantic-validated entry point (`MapOverlayOption`
        # only allows that model), but since the function is called
        # directly in these tests (and potentially from other Python code
        # bypassing schema validation), passing a bad/unexpected object
        # degrades silently to an empty layer list instead of erroring loudly.
        result = select_map_overlay(option=object())
        assert result == []
