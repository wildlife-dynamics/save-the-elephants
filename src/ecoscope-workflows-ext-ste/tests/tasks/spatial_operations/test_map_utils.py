"""Tests for ecoscope_workflows_ext_ste.tasks.spatial_operations._map_utils.

All three top-level functions (`combine_deckgl_map_layers`, `envelope_gdf`,
`compute_view_state_from_gdf`) are registered via `wt_registry.register()`,
which is a no-op at call time, so they behave as plain Python functions.

These tests assert on the *structure* of the returned `LayerDefinition` /
`ViewState` objects (dataclass / pydantic model field values) rather than
any rendered/visual output, since `combine_deckgl_map_layers` never touches
pydeck directly -- it only rearranges `LayerDefinition` objects.
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from ecoscope_workflows_ext_custom.tasks.results._map import (
    LegendSegment,
    LegendValue,
    ScatterplotLayerStyle,
    TextLayerStyle,
    ViewState,
    LayerDefinition,
)
from ecoscope_workflows_ext_ste.tasks.spatial_operations._map_utils import (
    combine_deckgl_map_layers,
    compute_view_state_from_gdf,
    envelope_gdf,
)


def _point_gdf(crs="EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs=crs)


def _make_layer(layer_type="ScatterplotLayer", legend=None, style=None) -> LayerDefinition:
    return LayerDefinition(
        layer_type=layer_type,
        layer_style=style or ScatterplotLayerStyle(),
        legend=legend,
        geodataframe=_point_gdf(),
    )


class TestCombineDeckglMapLayersEmptyInputs:
    def test_both_none_returns_empty_list(self):
        assert combine_deckgl_map_layers(None, None) == []

    def test_both_omitted_uses_defaults(self):
        assert combine_deckgl_map_layers() == []


class TestCombineDeckglMapLayersOrdering:
    def test_static_only_no_legend_preserves_input(self):
        layer_a = _make_layer("PathLayer")
        layer_b = _make_layer("ScatterplotLayer")
        result = combine_deckgl_map_layers(static_layers=[layer_a, layer_b], grouped_layers=None)
        assert result == [layer_a, layer_b]

    def test_grouped_only_returns_grouped_layers(self):
        layer_a = _make_layer("PathLayer")
        result = combine_deckgl_map_layers(static_layers=None, grouped_layers=[layer_a])
        assert result == [layer_a]

    def test_single_layer_not_wrapped_in_list_is_accepted(self):
        layer_a = _make_layer("PathLayer")
        result = combine_deckgl_map_layers(static_layers=layer_a, grouped_layers=None)
        assert result == [layer_a]

    def test_nested_lists_are_flattened(self):
        layer_a = _make_layer("PathLayer")
        layer_b = _make_layer("ScatterplotLayer")
        layer_c = _make_layer("PolygonLayer")
        result = combine_deckgl_map_layers(static_layers=[[layer_a, layer_b], layer_c], grouped_layers=None)
        assert result == [layer_a, layer_b, layer_c]

    def test_static_without_legend_renders_before_grouped(self):
        static_layer = _make_layer("PathLayer")  # no legend
        grouped_layer = _make_layer("ScatterplotLayer")
        result = combine_deckgl_map_layers(static_layers=[static_layer], grouped_layers=[grouped_layer])
        assert result == [static_layer, grouped_layer]

    def test_text_layers_always_rendered_last(self):
        text_layer = _make_layer("TextLayer", style=TextLayerStyle())
        path_layer = _make_layer("PathLayer")
        scatter_layer = _make_layer("ScatterplotLayer")
        result = combine_deckgl_map_layers(static_layers=[text_layer, path_layer], grouped_layers=[scatter_layer])
        assert result[-1] is text_layer
        assert result.index(text_layer) == len(result) - 1

    def test_multiple_text_layers_all_moved_to_end_preserving_relative_order(self):
        text_1 = _make_layer("TextLayer", style=TextLayerStyle())
        text_2 = _make_layer("TextLayer", style=TextLayerStyle())
        path_layer = _make_layer("PathLayer")
        result = combine_deckgl_map_layers(static_layers=[text_1, path_layer, text_2], grouped_layers=None)
        assert result == [path_layer, text_1, text_2]


class TestCombineDeckglMapLayersLegendCarrying:
    def test_static_legend_is_stripped_from_the_static_layer_itself(self):
        legend = LegendSegment(values=[LegendValue(label="A", color="#fff")], title="Boundary")
        static_layer = _make_layer("PolygonLayer", legend=legend)
        grouped_layer = _make_layer("ScatterplotLayer")
        result = combine_deckgl_map_layers(static_layers=[static_layer], grouped_layers=[grouped_layer])

        static_in_result = [layer for layer in result if layer.layer_type == "PolygonLayer"]
        assert len(static_in_result) == 1
        assert static_in_result[0].legend is None

    def test_static_legend_rides_on_a_phantom_copy_of_first_grouped_layer(self):
        legend = LegendSegment(values=[LegendValue(label="A", color="#fff")], title="Boundary")
        static_layer = _make_layer("PolygonLayer", legend=legend)
        grouped_layer = _make_layer("ScatterplotLayer")
        result = combine_deckgl_map_layers(static_layers=[static_layer], grouped_layers=[grouped_layer])

        carriers = [
            layer for layer in result if layer.layer_type == grouped_layer.layer_type and layer.legend is legend
        ]
        assert len(carriers) == 1
        # phantom carrier is a *copy* of the grouped layer, not the same object
        assert carriers[0] is not grouped_layer
        assert carriers[0].geodataframe is grouped_layer.geodataframe

    def test_legend_carrier_uses_first_grouped_layer_when_multiple_present(self):
        legend = LegendSegment(values=[LegendValue(label="A", color="#fff")], title="Boundary")
        static_layer = _make_layer("PolygonLayer", legend=legend)
        grouped_1 = _make_layer("ScatterplotLayer")
        grouped_2 = _make_layer("PathLayer")
        result = combine_deckgl_map_layers(static_layers=[static_layer], grouped_layers=[grouped_1, grouped_2])

        carrier_types = {layer.layer_type for layer in result if layer.legend is legend}
        assert carrier_types == {grouped_1.layer_type}

    def test_legend_lost_when_no_grouped_layers_to_carry_it(self):
        # documents current behavior: a static layer's legend has nowhere to
        # "ride along" on if there are no grouped layers at all, so it is
        # simply dropped rather than kept on the static layer.
        legend = LegendSegment(values=[LegendValue(label="A", color="#fff")], title="Boundary")
        static_layer = _make_layer("PolygonLayer", legend=legend)
        result = combine_deckgl_map_layers(static_layers=[static_layer], grouped_layers=None)
        assert len(result) == 1
        assert result[0].legend is None

    def test_static_layer_with_no_legend_is_unaffected_by_legend_carrying(self):
        static_layer = _make_layer("PolygonLayer", legend=None)
        grouped_layer = _make_layer("ScatterplotLayer")
        result = combine_deckgl_map_layers(static_layers=[static_layer], grouped_layers=[grouped_layer])
        # exactly static + grouped, no phantom carrier added
        assert result == [static_layer, grouped_layer]


class TestEnvelopeGdf:
    def test_default_expansion_factor_is_1_5(self, aois_gdf):
        default_result = envelope_gdf(aois_gdf)
        explicit_result = envelope_gdf(aois_gdf, expansion_factor=1.5)
        assert default_result.total_bounds == pytest.approx(explicit_result.total_bounds)

    def test_returns_single_row_polygon_geodataframe(self, aois_gdf):
        result = envelope_gdf(aois_gdf, expansion_factor=1.2)
        assert len(result) == 1
        assert result.geometry.iloc[0].geom_type == "Polygon"

    def test_preserves_input_crs(self, aois_gdf):
        result = envelope_gdf(aois_gdf, expansion_factor=1.2)
        assert result.crs == aois_gdf.crs

    def test_expansion_factor_1_matches_original_total_bounds(self, aois_gdf):
        result = envelope_gdf(aois_gdf, expansion_factor=1.0)
        assert result.total_bounds == pytest.approx(aois_gdf.total_bounds)

    def test_expansion_factor_2_doubles_width_and_height(self, aois_gdf):
        minx, miny, maxx, maxy = aois_gdf.total_bounds
        orig_width, orig_height = maxx - minx, maxy - miny

        result = envelope_gdf(aois_gdf, expansion_factor=2.0)
        rminx, rminy, rmaxx, rmaxy = result.total_bounds

        assert (rmaxx - rminx) == pytest.approx(orig_width * 2.0)
        assert (rmaxy - rminy) == pytest.approx(orig_height * 2.0)

    def test_expansion_is_centered_on_original_bbox_center(self, aois_gdf):
        minx, miny, maxx, maxy = aois_gdf.total_bounds
        center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2

        result = envelope_gdf(aois_gdf, expansion_factor=1.75)
        rminx, rminy, rmaxx, rmaxy = result.total_bounds
        assert (rminx + rmaxx) / 2 == pytest.approx(center_x)
        assert (rminy + rmaxy) / 2 == pytest.approx(center_y)

    @pytest.mark.parametrize("bad_factor", [0, -1, -0.5])
    def test_non_positive_expansion_factor_raises(self, aois_gdf, bad_factor):
        with pytest.raises(ValueError, match="expansion_factor must be greater than 0"):
            envelope_gdf(aois_gdf, expansion_factor=bad_factor)

    def test_single_polygon_input(self, kenya_pa_gdf):
        single = kenya_pa_gdf.iloc[[0]]
        result = envelope_gdf(single, expansion_factor=1.5)
        assert len(result) == 1
        assert result.crs == single.crs


class TestComputeViewStateFromGdf:
    def test_returns_view_state_with_center_within_bounds(self, aois_gdf):
        result = compute_view_state_from_gdf(aois_gdf)
        assert isinstance(result, ViewState)
        minx, miny, maxx, maxy = aois_gdf.total_bounds
        assert minx <= result.longitude <= maxx
        assert miny <= result.latitude <= maxy

    def test_zoom_is_within_default_bounds(self, aois_gdf):
        result = compute_view_state_from_gdf(aois_gdf)
        assert 0.0 <= result.zoom <= 18.0

    def test_pitch_and_bearing_pass_through(self, aois_gdf):
        result = compute_view_state_from_gdf(aois_gdf, pitch=30, bearing=-90)
        assert result.pitch == 30
        assert result.bearing == -90

    def test_default_pitch_and_bearing_are_zero(self, aois_gdf):
        result = compute_view_state_from_gdf(aois_gdf)
        assert result.pitch == 0
        assert result.bearing == 0

    def test_max_zoom_clamps_a_tiny_bbox(self):
        tiny = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(36.8, -1.3)], crs="EPSG:4326")
        result = compute_view_state_from_gdf(tiny, max_zoom=5.0)
        assert result.zoom == 5.0

    def test_larger_bbox_yields_lower_zoom_than_smaller_bbox(self):
        small = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Polygon([(36.0, -1.0), (36.1, -1.0), (36.1, -0.9), (36.0, -0.9)])],
            crs="EPSG:4326",
        )
        large = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Polygon([(30.0, -10.0), (41.0, -10.0), (41.0, 4.0), (30.0, 4.0)])],
            crs="EPSG:4326",
        )
        small_zoom = compute_view_state_from_gdf(small).zoom
        large_zoom = compute_view_state_from_gdf(large).zoom
        assert large_zoom < small_zoom

    def test_empty_gdf_raises_value_error(self):
        empty = gpd.GeoDataFrame({"id": []}, geometry=gpd.GeoSeries([], dtype="geometry"), crs="EPSG:4326")
        with pytest.raises(ValueError, match="GeoDataFrame is empty"):
            compute_view_state_from_gdf(empty)

    def test_projected_crs_input_is_reprojected_to_geographic(self, aois_gdf):
        projected = aois_gdf.to_crs("EPSG:32637")
        assert not projected.crs.is_geographic

        result_from_projected = compute_view_state_from_gdf(projected)
        result_from_geographic = compute_view_state_from_gdf(aois_gdf)

        assert result_from_projected.longitude == pytest.approx(result_from_geographic.longitude, abs=1e-6)
        assert result_from_projected.latitude == pytest.approx(result_from_geographic.latitude, abs=1e-6)
        assert result_from_projected.zoom == pytest.approx(result_from_geographic.zoom, abs=0.01)

    def test_missing_crs_raises_on_naive_geometry_reprojection(self):
        # SUSPECTED BUG: when `gdf.crs is None`, `compute_view_state_from_gdf`
        # unconditionally calls `gdf.to_crs("EPSG:4326")` (since `gdf.crs is
        # None` short-circuits the `or` before `.is_geographic` is checked).
        # geopandas raises a ValueError for "naive geometries" in this case
        # rather than the function giving a clear, intentional validation
        # message (contrast with `spatial_tag`, which explicitly checks for
        # `None` CRS and raises a clear error). This test documents the
        # *actual* current behavior; it is not asserting this is desired.
        no_crs = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(36.8, -1.3)])
        assert no_crs.crs is None
        with pytest.raises(ValueError, match="Cannot transform naive geometries"):
            compute_view_state_from_gdf(no_crs)
