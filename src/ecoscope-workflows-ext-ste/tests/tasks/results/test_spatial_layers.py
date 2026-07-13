from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from ecoscope_workflows_ext_custom.tasks.results._map import (
    GeoJSONLayerStyle,
    LayerDefinition,
    TextLayerStyle,
)

from ecoscope_workflows_ext_ste.tasks.results._spatial_layers import (
    CategoricalLayer,
    CustomPalette,
    create_categorical_layers,
    create_categorical_styled_layer,
    create_column_styled_layer,
    create_spatial_features_layer,
    geodataframe_from_layers,
)
from ecoscope_workflows_ext_ste.tasks.spatial_operations._spatial_features import (
    FeatureSetQuery,
)

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def kenyan_counties_gdf():
    """Polygon boundaries, no pre-existing style columns."""
    return gpd.read_file(TEST_DATA_DIR / "kenyan_counties.gpkg")


@pytest.fixture
def kenya_pa_gdf():
    """Protected areas with a categorical 'type' column
    (Community Conservancy / National Park / National Reserve)."""
    return gpd.read_file(TEST_DATA_DIR / "kenya_pa.gpkg")


@pytest.fixture
def aois_gdf():
    return gpd.read_file(TEST_DATA_DIR / "AOIs.gpkg", layer="AOIs")


# ========================================================================
# geodataframe_from_layers
# ========================================================================


def test_geodataframe_from_layers_combines_gdfs(kenyan_counties_gdf, aois_gdf):
    l1 = LayerDefinition(
        layer_type="GeoJsonLayer", layer_style=GeoJSONLayerStyle(), legend=None, geodataframe=kenyan_counties_gdf
    )
    l2 = LayerDefinition(layer_type="GeoJsonLayer", layer_style=GeoJSONLayerStyle(), legend=None, geodataframe=aois_gdf)

    result = geodataframe_from_layers(layers=[l1, l2])

    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == len(kenyan_counties_gdf) + len(aois_gdf)


def test_geodataframe_from_layers_single_layer(kenyan_counties_gdf):
    layer = LayerDefinition(
        layer_type="GeoJsonLayer", layer_style=GeoJSONLayerStyle(), legend=None, geodataframe=kenyan_counties_gdf
    )

    result = geodataframe_from_layers(layers=[layer])

    assert len(result) == len(kenyan_counties_gdf)


def test_geodataframe_from_layers_skips_text_layers(kenyan_counties_gdf):
    """TextLayer entries carry centroid geometry rather than the source
    feature, so they are excluded from the combined result."""
    text_layer = LayerDefinition(
        layer_type="TextLayer", layer_style=TextLayerStyle(), legend=None, geodataframe=kenyan_counties_gdf
    )

    result = geodataframe_from_layers(layers=[text_layer])

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.empty


def test_geodataframe_from_layers_skips_data_url_only_layers():
    """A layer backed only by a data_url has no in-memory geodataframe to extract."""
    layer = LayerDefinition(
        layer_type="GeoJsonLayer",
        layer_style=GeoJSONLayerStyle(),
        legend=None,
        data_url="https://example.com/data.geojson",
    )

    result = geodataframe_from_layers(layers=[layer])

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.empty


def test_geodataframe_from_layers_empty_list():
    result = geodataframe_from_layers(layers=[])

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.empty


# ========================================================================
# create_spatial_features_layer
# ========================================================================


def _styled_polygon_gdf(kenya_pa_gdf):
    styled = kenya_pa_gdf.copy()
    styled["get_fill_color"] = [[255, 0, 0, 255]] * len(styled)
    styled["get_line_color"] = [[0, 0, 0, 255]] * len(styled)
    styled["legend_title"] = "Protected Areas"
    styled["legend_label"] = styled["type"]
    return styled


def test_create_spatial_features_layer_geojson(kenya_pa_gdf):
    styled = _styled_polygon_gdf(kenya_pa_gdf)

    layers = create_spatial_features_layer(geodataframes=[styled])

    assert len(layers) == 1
    assert layers[0].layer_type == "GeoJsonLayer"
    assert len(layers[0].geodataframe) == len(kenya_pa_gdf)
    assert layers[0].legend.title == "Protected Areas"
    labels = {v.label for v in layers[0].legend.values}
    assert labels == set(kenya_pa_gdf["type"].unique())


def test_create_spatial_features_layer_icon_layer():
    icon_gdf = gpd.GeoDataFrame(
        {
            "icon_url": ["https://example.com/a.svg", "https://example.com/b.svg"],
            "icon_color": [[255, 0, 0, 255], [0, 255, 0, 255]],
            "legend_title": ["Markers", "Markers"],
            "legend_label": ["A", "B"],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        crs="EPSG:4326",
    )

    layers = create_spatial_features_layer(geodataframes=[icon_gdf])

    assert len(layers) == 1
    assert layers[0].layer_type == "IconLayer"
    assert {v.label for v in layers[0].legend.values} == {"A", "B"}


def test_create_spatial_features_layer_mixed_icon_and_geometry(kenya_pa_gdf):
    """A single source containing both icon rows and plain geometry rows
    produces one GeoJsonLayer and one IconLayer, sharing a single legend."""
    styled = _styled_polygon_gdf(kenya_pa_gdf.head(3))
    styled["icon_url"] = [None, "https://example.com/a.svg", None]

    layers = create_spatial_features_layer(geodataframes=[styled])

    assert {layer.layer_type for layer in layers} == {"GeoJsonLayer", "IconLayer"}
    assert len(layers[0].geodataframe) == 2  # non-icon rows
    assert len(layers[1].geodataframe) == 1  # icon row


def test_create_spatial_features_layer_backcompat_single_gdf(kenya_pa_gdf):
    """A bare GeoDataFrame (not wrapped in a list) is accepted for backwards compatibility."""
    styled = _styled_polygon_gdf(kenya_pa_gdf)

    layers = create_spatial_features_layer(geodataframes=styled)

    assert len(layers) == 1


def test_create_spatial_features_layer_skips_empty_sources(kenya_pa_gdf):
    styled = _styled_polygon_gdf(kenya_pa_gdf)
    empty_gdf = gpd.GeoDataFrame()

    layers = create_spatial_features_layer(geodataframes=[empty_gdf, styled])

    assert len(layers) == 1


def test_create_spatial_features_layer_empty_input():
    assert create_spatial_features_layer(geodataframes=[]) == []
    assert create_spatial_features_layer(geodataframes=None) == []


# ========================================================================
# create_column_styled_layer
# ========================================================================


def _hex_colored_gdf(kenya_pa_gdf):
    gdf = kenya_pa_gdf.copy()
    gdf["type_hex_colors"] = gdf["type"].map(
        {
            "National Park": "#115631",
            "National Reserve": "#a6b697",
            "Community Conservancy": "#e63946",
        }
    )
    return gdf


def test_create_column_styled_layer_basic(kenya_pa_gdf):
    gdf = _hex_colored_gdf(kenya_pa_gdf)

    layers = create_column_styled_layer(
        geodataframes=[gdf], fill_color_col="type_hex_colors", group_by="type", legend_title="Protected Areas"
    )

    assert len(layers) == 1
    assert layers[0].layer_type == "GeoJsonLayer"
    assert layers[0].legend.title == "Protected Areas"
    assert len(layers[0].legend.values) == 3
    assert len(layers[0].geodataframe) == len(kenya_pa_gdf)


def test_create_column_styled_layer_group_by_defaults_to_geom_type(kenya_pa_gdf):
    """group_by defaults to 'type_name', which isn't a column here, so the
    legend falls back to grouping by geometry type."""
    gdf = _hex_colored_gdf(kenya_pa_gdf)

    layers = create_column_styled_layer(geodataframes=[gdf], fill_color_col="type_hex_colors")

    assert len(layers) == 1
    assert [v.label for v in layers[0].legend.values] == ["MultiPolygon"]


def test_create_column_styled_layer_backcompat_single_gdf(kenya_pa_gdf):
    gdf = _hex_colored_gdf(kenya_pa_gdf)

    layers = create_column_styled_layer(geodataframes=gdf, fill_color_col="type_hex_colors")

    assert len(layers) == 1


def test_create_column_styled_layer_skips_empty_and_none():
    empty_gdf = gpd.GeoDataFrame()

    assert create_column_styled_layer(geodataframes=[empty_gdf], fill_color_col="x") == []
    assert create_column_styled_layer(geodataframes=[None], fill_color_col="x") == []
    assert create_column_styled_layer(geodataframes=[], fill_color_col="x") == []


# ========================================================================
# create_categorical_layers / CategoricalLayer
# ========================================================================


def test_create_categorical_layers_auto_palette(kenya_pa_gdf):
    spec = CategoricalLayer(file_path=str(TEST_DATA_DIR / "kenya_pa.gpkg"), color_by="type")

    layers = create_categorical_layers(layers=[spec])

    assert len(layers) == 1
    labels = {v.label for v in layers[0].legend.values}
    assert labels == set(kenya_pa_gdf["type"].unique())
    # every category got a distinct auto-assigned color
    colors = [v.color for v in layers[0].legend.values]
    assert len(set(colors)) == len(colors)


def test_create_categorical_layers_explicit_color_mapping(kenya_pa_gdf):
    spec = CategoricalLayer(
        file_path=str(TEST_DATA_DIR / "kenya_pa.gpkg"),
        color_by="type",
        color_mapping={"National Park": "#115631"},
        unmapped="#808080",
    )

    layers = create_categorical_layers(layers=[spec])

    values = {v.label: v.color for v in layers[0].legend.values}
    assert values["National Park"] == "rgba(17, 86, 49, 1.0)"
    assert values["National Reserve"] == "rgba(128, 128, 128, 1.0)"
    assert values["Community Conservancy"] == "rgba(128, 128, 128, 1.0)"


def test_create_categorical_layers_unmapped_drop(kenya_pa_gdf):
    spec = CategoricalLayer(
        file_path=str(TEST_DATA_DIR / "kenya_pa.gpkg"),
        color_by="type",
        color_mapping={"National Park": "#115631"},
        unmapped="drop",
    )

    layers = create_categorical_layers(layers=[spec])

    assert len(layers) == 1
    assert [v.label for v in layers[0].legend.values] == ["National Park"]
    expected_count = (kenya_pa_gdf["type"] == "National Park").sum()
    assert len(layers[0].geodataframe) == expected_count


def test_create_categorical_layers_custom_palette(kenya_pa_gdf):
    spec = CategoricalLayer(
        file_path=str(TEST_DATA_DIR / "kenya_pa.gpkg"),
        color_by="type",
        palette=CustomPalette(colors=["#e63946", "#457b9d"]),
    )

    layers = create_categorical_layers(layers=[spec])

    colors = {v.color for v in layers[0].legend.values}
    # only 2 colors in the palette but 3 categories -> the palette cycles
    assert colors <= {"rgba(230, 57, 70, 1.0)", "rgba(69, 123, 157, 1.0)"}


def test_categorical_layer_invalid_unmapped_raises():
    with pytest.raises(Exception, match="unmapped"):
        CategoricalLayer(file_path=str(TEST_DATA_DIR / "kenya_pa.gpkg"), color_by="type", unmapped="notavalid")


def test_categorical_layer_requires_exactly_one_source():
    with pytest.raises(Exception, match="Provide exactly one source"):
        CategoricalLayer(color_by="type")

    with pytest.raises(Exception, match="Provide exactly one source"):
        CategoricalLayer(
            file_path=str(TEST_DATA_DIR / "kenya_pa.gpkg"),
            query=FeatureSetQuery(featureset_name="Boundaries"),
            color_by="type",
        )


def test_create_categorical_layers_query_without_client_raises():
    spec = CategoricalLayer(query=FeatureSetQuery(featureset_name="Boundaries"), color_by="type")

    with pytest.raises(ValueError, match="EarthRanger `client` is required"):
        create_categorical_layers(layers=[spec], client=None)


def test_create_categorical_layers_missing_file_raises():
    spec = CategoricalLayer(file_path=str(TEST_DATA_DIR / "does_not_exist.gpkg"), color_by="type")

    with pytest.raises(RuntimeError):
        create_categorical_layers(layers=[spec])


# ========================================================================
# create_categorical_styled_layer
# ========================================================================


def test_create_categorical_styled_layer_basic(kenya_pa_gdf):
    layers = create_categorical_styled_layer(geodataframe=kenya_pa_gdf, color_by="type")

    assert len(layers) == 1
    assert layers[0].layer_type == "GeoJsonLayer"
    labels = {v.label for v in layers[0].legend.values}
    assert labels == set(kenya_pa_gdf["type"].unique())


def test_create_categorical_styled_layer_missing_column_raises(kenya_pa_gdf):
    with pytest.raises(KeyError, match="nonexistent"):
        create_categorical_styled_layer(geodataframe=kenya_pa_gdf, color_by="nonexistent")


def test_create_categorical_styled_layer_empty_gdf_returns_empty_list():
    empty_gdf = gpd.GeoDataFrame()

    assert create_categorical_styled_layer(geodataframe=empty_gdf, color_by="type") == []


def test_create_categorical_styled_layer_dict_palette_bypass(kenya_pa_gdf):
    """Direct (non-workflow) calls may pass a raw palette dict instead of a
    validated ColorPalette model; it should still be accepted."""
    layers = create_categorical_styled_layer(
        geodataframe=kenya_pa_gdf,
        color_by="type",
        palette={"type_": "custom", "colors": ["#111111"]},
    )

    colors = {v.color for v in layers[0].legend.values}
    assert colors == {"rgba(17, 17, 17, 1.0)"}


def test_create_categorical_styled_layer_invalid_unmapped_raises(kenya_pa_gdf):
    with pytest.raises(ValueError, match="unmapped"):
        create_categorical_styled_layer(geodataframe=kenya_pa_gdf, color_by="type", unmapped="bogus")


def test_create_categorical_styled_layer_group_by_defaults_to_color_by(kenya_pa_gdf):
    """group_by='' should fall back to color_by for legend grouping."""
    layers = create_categorical_styled_layer(geodataframe=kenya_pa_gdf, color_by="type", group_by="")

    labels = {v.label for v in layers[0].legend.values}
    assert labels == set(kenya_pa_gdf["type"].unique())
