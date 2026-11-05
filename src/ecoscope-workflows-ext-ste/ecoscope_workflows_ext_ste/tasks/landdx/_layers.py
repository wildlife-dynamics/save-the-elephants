from wt_registry import register
import geopandas as gpd
from typing import Annotated, List, Union
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema
from ecoscope.platform.connections import EarthRangerClient
from ecoscope.platform.tasks.results._pydeck import (
    TextLayerStyle,
)
from ecoscope.platform.tasks.results._pydeck import (
    PydeckLayerDefinition as LayerDefinition,
)
from ecoscope_workflows_ext_custom.tasks.io import load_df
from ..spatial_operations._spatial_features import (
    LayerStyle,
    _apply_geo_style,
)
from ..results._spatial_layers import (
    _layers_for_gdf,
    create_categorical_styled_layer,
)
from ..io import load_mapbook_file


_LDX_COLOR_MAPPING = {
    "Community Conservancy": "#a6b697",
    "National Reserve": "#88a78e",
    "National Park": "#115631",
}

_LDX_FILE = "mapbook_ldx_layers.gpkg"


def _build_ldx_layers(gdf: gpd.GeoDataFrame) -> List[LayerDefinition]:
    """Style the LandDx layers into LayerDefinitions."""
    gdf = gdf[gdf["type"].isin(_LDX_COLOR_MAPPING)][["type", "name", "geometry"]].copy()
    if gdf.empty:
        return []

    layers = create_categorical_styled_layer(
        geodataframe=gdf,
        color_by="type",
        color_mapping=_LDX_COLOR_MAPPING,
        unmapped="drop",
        fill_opacity=0.35,
        line_opacity=0.35,
        line_width=2.25,
        legend_title="Land Use",
    )

    text_gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid].copy()
    text_gdf = text_gdf.to_crs("EPSG:3857")
    text_gdf["geometry"] = text_gdf.geometry.centroid
    text_gdf = text_gdf.to_crs("EPSG:4326")
    layers.append(
        LayerDefinition(
            layer_type="TextLayer",
            geodataframe=text_gdf,
            layer_style=TextLayerStyle(
                get_text="name",
                get_color=[20, 20, 20, 255],
                get_size=1000,
                size_units="meters",
                size_min_pixels=40,
                size_max_pixels=75,
                size_scale=1.25,
                font_family="Arial",
                font_weight="normal",
                get_text_anchor="middle",
                get_alignment_baseline="center",
                billboard=True,
                background_padding=[4, 8],
                pickable=True,
                auto_highlight=False,
            ),
            legend=None,
        )
    )
    return layers


class LandDxOverlayOption(BaseModel):
    """Use the default LandDx protected-areas layer."""

    model_config = ConfigDict(title="LandDx (Default Protected Areas)")


class ERSpatialFeatureGroupOverlayOption(BaseModel):
    """Overlay one or more EarthRanger spatial feature groups."""

    model_config = ConfigDict(title="EarthRanger Spatial Feature Groups")

    # A plain list of group names so ecoscope-rjsf's EarthRangerEnumResolver
    # (type: spatial_feature_group) can turn it into a drop-down populated from
    # the selected data source; see the rjsf-overrides in the workflow specs.
    spatial_feature_groups: Annotated[
        list[str],
        Field(
            min_length=1,
            title="Spatial Feature Groups",
            description="EarthRanger spatial feature groups to display on the maps.",
        ),
    ]
    style: Annotated[
        list[LayerStyle],
        Field(
            default=[],
            max_length=1,
            title="Style",
            description="Optional: override how the features are rendered. "
            "Leave empty to keep the colours set in EarthRanger.",
        ),
    ] = []


MapOverlayOption = Union[ERSpatialFeatureGroupOverlayOption, LandDxOverlayOption]


def _build_er_group_layers(
    client: EarthRangerClient,
    group_names: list[str],
    style: LayerStyle,
) -> List[LayerDefinition]:
    """Fetch each spatial feature group and style it into LayerDefinitions.

    Each group gets its own legend segment titled with the group name. Groups
    with no features are skipped.
    """
    server_url = getattr(client, "server", "").rstrip("/")
    layers: List[LayerDefinition] = []
    for name in group_names:
        gdf = client.get_spatial_features_group(
            spatial_features_group_name=name,
            spatial_features_group_id=None,
            with_group_data=False,
        )
        if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
            continue
        styled = _apply_geo_style(
            gdf.reset_index(drop=True),
            style,
            group_by="title",
            legend_title=name,
            server_url=server_url,
        )
        layers.extend(_layers_for_gdf(styled))  # type: ignore[arg-type]
    return layers


# Tagged as IO because the EarthRanger option calls EarthRanger. In mock-IO test
# runs the connection task returns only the connection name, so running this task
# for real would try to log in to EarthRanger while validating `client`. Mock-IO
# runs use select-map-overlay.example-return.json (no overlay layers) instead.
@register(tags=["io"])
def select_map_overlay(
    option: Annotated[
        MapOverlayOption,
        Field(description="Select the overlay source to display on maps."),
    ],
    client: Annotated[
        EarthRangerClient | SkipJsonSchema[None],
        Field(
            default=None,
            description="EarthRanger client, required only for the EarthRanger option.",
            exclude=True,
        ),
    ] = None,
    ldx_path: Annotated[
        str | SkipJsonSchema[None],
        Field(
            default=None,
            description="Path to the LandDx GeoPackage (local, file:// or gs://). "
            "Defaults to the copy shipped with ext-ste.",
            exclude=True,
        ),
    ] = None,
) -> Annotated[List[LayerDefinition], Field()]:
    """Return map overlay layers from the selected source.
    - **EarthRanger Spatial Feature Groups** – fetches each named spatial feature
      group from EarthRanger and renders it with its EarthRanger styling, or the
      optional ``style`` override.
    - **LandDx** – loads the LandDx GeoPackage shipped with ext-ste (or ``ldx_path``)
      and styles it by land-use type using ``create_categorical_styled_layer``.

    """
    if isinstance(option, LandDxOverlayOption):
        gdf = load_df(file_path=ldx_path or load_mapbook_file(_LDX_FILE))
        return _build_ldx_layers(gdf)

    if isinstance(option, ERSpatialFeatureGroupOverlayOption):
        if client is None:
            raise ValueError("An EarthRanger `client` is required for the EarthRanger overlay option.")
        style = option.style[0] if option.style else LayerStyle()
        return _build_er_group_layers(client, option.spatial_feature_groups, style)
    return []
