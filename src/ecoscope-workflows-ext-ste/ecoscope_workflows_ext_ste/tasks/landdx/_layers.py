from wt_registry import register
import geopandas as gpd
from typing import Annotated, List, Union
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema
from ecoscope.platform.connections import EarthRangerClient
from ecoscope_workflows_ext_custom.tasks.results._map import (
    TextLayerStyle,
    LayerDefinition,
)
from ..spatial_operations._spatial_features import (
    LocalSpatialLayer,
    SpatialFeatureLayer,
)
from ..results._spatial_layers import (
    _layers_for_gdf,
    create_categorical_styled_layer,
)
from ..io import fetch_and_persist_file


_LDX_COLOR_MAPPING = {
    "Community Conservancy": "#a6b697",
    "National Reserve": "#88a78e",
    "National Park": "#115631",
}

_LDX_URL = (
    "https://www.dropbox.com/scl/fi/uitptfgxk4wnfcnv9k96a/mapbook_ldx_layers.gpkg"
    "?rlkey=xi2azbfzqix9udytv3smsf6eh&st=249w3d2x&dl=0"
)


def _build_ldx_layers(file_path: str) -> List[LayerDefinition]:
    """Load and style the LandDx GeoPackage into LayerDefinitions."""
    gdf = gpd.read_file(file_path)
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


class ERSpatialFeatureOverlayOption(BaseModel):
    """Overlay one or more EarthRanger spatial feature layers."""

    model_config = ConfigDict(title="EarthRanger Spatial Feature")

    layers: Annotated[
        list[SpatialFeatureLayer],
        Field(
            description="One or more EarthRanger feature layers. Each entry specifies "
            "what to load (featureset / type / id) and optional style overrides.",
        ),
    ]


class LocalFileOverlayOption(BaseModel):
    """Overlay features loaded from one or more local geospatial files."""

    model_config = ConfigDict(title="Custom Local File")

    files: Annotated[
        list[LocalSpatialLayer],
        Field(
            description="One or more local GeoJSON / GeoPackage / GeoParquet files. "
            "Each file (or split group within a file) becomes its own styled layer.",
        ),
    ]


MapOverlayOption = Union[LandDxOverlayOption, ERSpatialFeatureOverlayOption, LocalFileOverlayOption]


@register()
def select_map_overlay(
    option: Annotated[
        MapOverlayOption,
        Field(description="Select the overlay source to display on maps."),
    ],
    client: Annotated[
        EarthRangerClient | SkipJsonSchema[None],
        Field(
            default=None,
            description="EarthRanger client — required only for the EarthRanger option.",
            exclude=True,
        ),
    ] = None,
    output_dir: Annotated[
        str,
        Field(
            default="/tmp",
            description="Directory used to cache downloaded files (LandDx option).",
        ),
    ] = "/tmp",
    ldx_url: Annotated[
        str,
        Field(
            description="URL to download the LandDx GeoPackage file.",
            exclude=True,
        ),
    ] = _LDX_URL,
) -> Annotated[List[LayerDefinition], Field()]:
    """Return map overlay layers from the selected source.

    - **LandDx** – downloads the standard LandDx GeoPackage and styles it by
      land-use type using ``create_categorical_styled_layer``.
    - **EarthRanger Spatial Feature** – fetches one or more EarthRanger feature
      layers using the SpatialFeatureLayer query model (featureset / type / id).
    - **Custom Local File** – loads one or more local GeoJSON / GeoPackage /
      GeoParquet files with optional per-layer styling.
    """
    if isinstance(option, LandDxOverlayOption):
        file_path = fetch_and_persist_file(
            url=ldx_url,
            output_path=output_dir,
            overwrite_existing=False,
            unzip=False,
            retries=2,
        )
        return _build_ldx_layers(file_path)

    if isinstance(option, ERSpatialFeatureOverlayOption):
        if client is None:
            raise ValueError("An EarthRanger `client` is required for the EarthRanger overlay option.")
        server_url = getattr(client, "server", "").rstrip("/")
        layers: List[LayerDefinition] = []
        for spec in option.layers:
            spec = SpatialFeatureLayer.model_validate(spec)
            gdf = spec.load(client, server_url)
            if gdf.empty:
                continue
            layers.extend(_layers_for_gdf(gdf))  # type: ignore[arg-type]
        return layers

    if isinstance(option, LocalFileOverlayOption):
        layers = []
        for spec in option.files:
            spec = LocalSpatialLayer.model_validate(spec)
            for gdf in spec.load():
                layers.extend(_layers_for_gdf(gdf))  # type: ignore[arg-type]
        return layers

    return []
