import math
import geopandas as gpd
from pydantic import Field
from dataclasses import replace
from shapely.geometry import box
from wt_registry import register
from typing import Union, Optional, Annotated, List
from ecoscope_workflows_ext_custom.tasks.results._map import (
    TextLayerStyle,
    LayerDefinition,
    ViewState,
)
from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame


def _flatten(layers) -> list[LayerDefinition]:
    if layers is None:
        return []
    if not isinstance(layers, list):
        return [layers]
    flat = []
    for item in layers:
        flat.extend(_flatten(item) if isinstance(item, list) else [item])
    return flat


@register()
def combine_deckgl_map_layers(
    static_layers: Annotated[
        Optional[Union[LayerDefinition, List[Union[LayerDefinition, List[LayerDefinition]]]]],
        Field(description="Static layers (e.g., base maps, boundaries)."),
    ] = None,
    grouped_layers: Annotated[
        Optional[Union[LayerDefinition, List[Union[LayerDefinition, List[LayerDefinition]]]]],
        Field(description="Grouped layers generated from split data."),
    ] = None,
) -> list[LayerDefinition]:
    """
    Combine static and grouped map layers into a render-ready list.

    Static layers render below grouped layers. Their legends ride along on
    "phantom" copies of the first grouped layer so they appear in the
    legend panel without forcing the static layer's own geometry to render
    on top. Text-styled layers are moved to the end so they render on top.

    Returns an empty list if both inputs are empty.
    """
    flat_static = _flatten(static_layers)
    flat_grouped = _flatten(grouped_layers)
    print(f"[combine_layers] Combining {len(flat_static)} static + {len(flat_grouped)} grouped layers")

    static_with_legend = [layer for layer in flat_static if layer.legend is not None]
    static_without_legend = [layer for layer in flat_static if layer.legend is None]

    static_stripped = [replace(layer, legend=None) for layer in static_with_legend]
    legend_carriers = (
        [replace(flat_grouped[0], legend=static.legend) for static in static_with_legend] if flat_grouped else []
    )

    all_layers = static_without_legend + static_stripped + flat_grouped + legend_carriers

    # Text layers always render on top.
    text = [layer for layer in all_layers if isinstance(layer.layer_style, TextLayerStyle)]
    other = [layer for layer in all_layers if not isinstance(layer.layer_style, TextLayerStyle)]
    result = other + text
    print(f"[combine_layers] Final stack: {len(result)} layers ({len(other)} base, {len(text)} text on top)")
    return result


@register()
def envelope_gdf(
    gdf: Annotated[
        AnyGeoDataFrame,
        Field(description="Input GeoDataFrame to create envelope from"),
    ],
    expansion_factor: Annotated[
        float,
        Field(description="Factor to expand the bounding box (e.g., 1.2 = 20% larger)"),
    ] = 1.50,
) -> Annotated[
    AnyGeoDataFrame,
    Field(description="GeoDataFrame containing the expanded envelope/bounding box"),
]:
    """
    Create an expanded envelope (bounding box) around all geometries in a GeoDataFrame.

    Args:
        gdf: Input GeoDataFrame
        expansion_factor: Multiplier for expanding the bounding box (> 0)
            - 1.0 = no expansion
            - 1.2 = 20% larger
            - 0.8 = 20% smaller

    Returns:
        GeoDataFrame with a single polygon representing the expanded envelope
    """
    print(f"[envelope_gdf] Creating {expansion_factor}x expanded bounding box for {len(gdf)} features")
    if expansion_factor <= 0:
        raise ValueError("expansion_factor must be greater than 0")

    envelope = gdf.union_all().envelope
    minx, miny, maxx, maxy = envelope.bounds

    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    width = maxx - minx
    height = maxy - miny

    new_width = width * expansion_factor
    new_height = height * expansion_factor

    new_minx = center_x - new_width / 2
    new_maxx = center_x + new_width / 2
    new_miny = center_y - new_height / 2
    new_maxy = center_y + new_height / 2

    expanded_envelope = box(new_minx, new_miny, new_maxx, new_maxy)
    envelope_gdf = gpd.GeoDataFrame({"geometry": [expanded_envelope]}, crs=gdf.crs)

    print(f"[envelope_gdf] Envelope bounds: ({new_minx:.4f}, {new_miny:.4f}) to ({new_maxx:.4f}, {new_maxy:.4f})")
    return envelope_gdf


def _zoom_from_bbox(
    minx,
    miny,
    maxx,
    maxy,
    map_width_px=800,
    map_height_px=600,
    tile_size=512,  # matches the deck.gl / MapLibre zoom convention
    min_zoom=0.0,
    max_zoom=18.0,  # cap at the basemap's max available zoom
) -> float:
    """
    Zoom level to fit a bbox, clamped so we never request a zoom the
    basemap can't serve. Coordinates must be EPSG:4326.
    """
    width_deg = abs(maxx - minx)
    height_deg = abs(maxy - miny)
    center_lat = (miny + maxy) / 2.0

    height_km = height_deg * 111.0
    width_km = width_deg * 111.0 * abs(math.cos(math.radians(center_lat)))

    world_km = 40075.0
    zooms = []
    if width_km > 1e-9:
        zooms.append(math.log2(world_km * map_width_px / (tile_size * width_km)))
    if height_km > 1e-9:
        zooms.append(math.log2(world_km * map_height_px / (tile_size * height_km)))

    # No extent at all (single point) -> zoom in as far as the basemap allows.
    zoom = min(zooms) if zooms else max_zoom
    zoom = max(min_zoom, min(max_zoom, zoom))
    return round(zoom, 2)


@register()
def compute_view_state_from_gdf(
    gdf,
    pitch: Annotated[int, AdvancedField(default=0, ge=0, le=90, description="...")] = 0,
    bearing: Annotated[int, AdvancedField(default=0, ge=-180, le=180, description="...")] = 0,
    max_zoom: Annotated[
        float,
        AdvancedField(
            default=18.0,
            ge=0,
            le=24,
            description="Highest zoom to allow. Set to the basemap tile layer's "
            "max available zoom so the view never requests tiles that don't exist.",
        ),
    ] = 18.0,
) -> ViewState:
    print(f"[view_state] Computing map view for {len(gdf)} features (CRS: {gdf.crs})")
    if gdf.empty:
        raise ValueError("GeoDataFrame is empty. Cannot compute ViewState.")

    if gdf.crs is None or not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lon = (minx + maxx) / 2.0
    center_lat = (miny + maxy) / 2.0
    zoom = _zoom_from_bbox(minx, miny, maxx, maxy, max_zoom=max_zoom)

    result = ViewState(
        longitude=center_lon,
        latitude=center_lat,
        zoom=zoom,
        pitch=pitch,
        bearing=bearing,
    )
    print(f"[view_state] Centered at ({center_lat:.4f}°N, {center_lon:.4f}°E), zoom={zoom}")
    return result
