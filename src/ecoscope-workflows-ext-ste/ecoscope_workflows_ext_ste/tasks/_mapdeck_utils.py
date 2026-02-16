import math
import logging
from enum import Enum
from pydantic import Field
from typing import TypedDict, Literal, Tuple
from pydantic.json_schema import SkipJsonSchema
from ecoscope_workflows_core.decorators import task
from typing import Annotated, Union, List, Dict, Optional
from ecoscope_workflows_core.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope_workflows_ext_custom.tasks.results._map import (
    LegendDefinition,
    PathLayerStyle,
    GeoJSONLayerStyle,
    ScatterplotLayerStyle,
    PolygonLayerStyle,
    TextLayerStyle,
    IconLayerStyle,
    ViewState,
    LayerDefinition,
    create_path_layer,
    create_geojson_layer,
    create_scatterplot_layer,
)
from shapely.geometry import box
import geopandas as gpd

logger = logging.getLogger(__name__)


class SupportedFormat(str, Enum):
    GPKG = ".gpkg"
    GEOJSON = ".geojson"
    SHP = ".shp"


class GdfWithGeometryType(TypedDict):
    gdf: AnyGeoDataFrame
    geometry_type: str


SUPPORTED_FORMATS = [f.value for f in SupportedFormat]


class DeckGLStyleParams(TypedDict, total=False):
    """Universal DeckGL style parameters that can be used for any layer type."""

    # Common parameters
    auto_highlight: bool
    opacity: float
    pickable: bool

    # Position and geometry accessors
    get_position: str
    get_path: str
    get_polygon: str

    # Color accessors
    get_color: str | list[int] | list[list[int]] | None
    get_fill_color: str | list[int] | list[list[int]] | None
    get_line_color: str | list[int] | list[list[int]] | None
    get_background_color: str | list[int] | list[list[int]] | None
    get_border_color: str | list[int] | list[list[int]] | None
    color_column: str | None
    fill_color_column: str | None

    # Dimension accessors
    get_width: float | None
    get_line_width: float | None
    get_border_width: float | None
    get_radius: float | None
    get_point_radius: float | None
    get_elevation: float | None
    get_size: int | str | float
    get_angle: float | None

    # Weight and aggregation
    get_color_weight: float | None
    color_aggregation: str | None
    elevation_aggregation: str | None

    # Width styling
    width_scale: float
    width_min_pixels: float
    width_max_pixels: float | None
    width_units: Literal["pixels", "meters"]

    # Line width styling
    line_width_units: Literal["pixels", "meters"]
    line_width_scale: float
    line_width_min_pixels: float
    line_width_max_pixels: float | None
    line_miter_limit: float
    line_joint_rounded: bool

    # Radius styling
    radius: float
    radius_units: Literal["pixels", "meters"]
    radius_scale: float
    radius_min_pixels: float
    radius_max_pixels: float | None

    # Point radius styling
    point_radius_units: Literal["pixels", "meters"]
    point_radius_scale: float
    point_radius_min_pixels: float
    point_radius_max_pixels: float | None

    # Size styling
    size_scale: float | int
    size_units: Literal["pixels", "meters"]
    size_min_pixels: float
    size_max_pixels: float | None

    # Elevation styling
    elevation_scale: float

    # Fill and stroke
    filled: bool
    stroked: bool
    extruded: bool
    wireframe: bool
    antialiasing: bool

    # Line caps and joints
    cap_rounded: bool
    joint_rounded: bool
    rounded: bool

    # Coverage (for hexagon layer)
    coverage: float

    # Rendering
    billboard: bool

    # Text layer specific
    get_text: str
    background: bool
    background_border_radius: float | Tuple[float, float, float, float] | None
    background_padding: Tuple[float, float] | Tuple[float, float, float, float] | None
    font_family: str
    font_weight: str | None
    line_height: float | None
    outline_width: float | None
    outline_color: Tuple[float, float, float, float] | None
    word_break: Literal["break-all", "break-word"] | None
    max_width: float | None
    get_text_anchor: Literal["start", "middle", "end"]
    get_alignment_baseline: Literal["top", "center", "bottom"]
    get_pixel_offset: Tuple[float, float] | None

    # Icon layer specific
    get_icon: str


def _zoom_from_bbox(minx, miny, maxx, maxy, map_width_px=800, map_height_px=600) -> float:
    """
    Calculate zoom level to fit bounding box in a given map size.
    This approach considers both dimensions for optimal fitting.

    Args:
        minx, miny, maxx, maxy (float): bounding box coordinates, must be in EPSG:4326
        map_width_px, map_height_px (int): target map dimensions in pixels

    Returns:
        float: zoom level that fits the bbox in the map
    """
    width_deg = abs(maxx - minx)
    height_deg = abs(maxy - miny)
    center_lat = (miny + maxy) / 2

    # Convert to km
    height_km = height_deg * 111.0
    width_km = width_deg * 111.0 * abs(math.cos(math.radians(center_lat)))

    world_width_km = 40075
    world_height_km = 40075

    zoom_for_width = math.log2(world_width_km * map_width_px / (512 * width_km))
    zoom_for_height = math.log2(world_height_km * map_height_px / (512 * height_km))

    zoom = min(zoom_for_width, zoom_for_height)
    zoom = round(max(0, min(20, zoom)), 2)
    return zoom


@task
def view_state_deck_gdf(
    gdf,
    pitch: int = 0,
    bearing: int = 0,
) -> ViewState:
    if gdf.empty:
        raise ValueError("GeoDataFrame is empty. Cannot compute ViewState.")

    if gdf.crs is None or not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lon = (minx + maxx) / 2.0
    center_lat = (miny + maxy) / 2.0
    zoom = _zoom_from_bbox(minx, miny, maxx, maxy)
    return ViewState(longitude=center_lon, latitude=center_lat, zoom=zoom, pitch=pitch, bearing=bearing)


@task
def get_image_zoom_value(
    gdf,
) -> float:
    if gdf.empty:
        raise ValueError("GeoDataFrame is empty. Cannot compute ViewState.")

    if gdf.crs is None or not gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = gdf.total_bounds
    zoom = _zoom_from_bbox(minx, miny, maxx, maxy)
    return zoom


@task
def custom_view_state_from_gdf(
    gdf: AnyGeoDataFrame,
    max_zoom: float = 20,
) -> Annotated[ViewState, Field()]:
    import pydeck as pdk

    if gdf is None or gdf.empty:
        return ViewState()

    gdf = gdf.to_crs(epsg=4326)
    bounds = gdf.total_bounds

    bbox = [
        [bounds[0], bounds[1]],
        [bounds[2], bounds[3]],
    ]

    computed_zoom = pdk.data_utils.viewport_helpers.bbox_to_zoom_level(bbox)
    centerLon = (bounds[0] + bounds[2]) / 2
    centerLat = (bounds[1] + bounds[3]) / 2

    return ViewState(longitude=centerLon, latitude=centerLat, zoom=min(max_zoom, computed_zoom))


@task
def combine_deckgl_map_layers(
    static_layers: Annotated[
        Union[LayerDefinition, List[LayerDefinition | List[LayerDefinition]]],
        Field(description="Static layers from local files or base maps."),
    ] = None,
    grouped_layers: Annotated[
        Union[LayerDefinition, List[LayerDefinition | List[LayerDefinition]]],
        Field(description="Grouped layers generated from split/grouped data."),
    ] = None,
) -> list[LayerDefinition]:
    """
    Combine static and grouped map layers into a single list for rendering in `draw_ecomap`.
    Automatically flattens nested lists to handle cases where layer generation tasks return lists.
    Static layer legends are merged into grouped layers to appear in legend while static layers
    render at the bottom of the map.
    """

    def flatten_layers(layers):
        """Recursively flatten nested lists of LayerDefinition objects."""
        if not isinstance(layers, list):
            return [layers]
        flattened = []
        for item in layers:
            if isinstance(item, list):
                flattened.extend(flatten_layers(item))
            else:
                flattened.append(item)
        return flattened

    flat_static = flatten_layers(static_layers) if static_layers else []
    flat_grouped = flatten_layers(grouped_layers) if grouped_layers else []
    flat_static_legend = [i for i in flat_static if i.legend]
    flat_static_no_legend = [i for i in flat_static if i.legend is None]

    # Extract legends from static layers and create new grouped layer copies with those legends
    grouped_with_static_legends = []
    for static_layer in flat_static_legend:
        # Create a copy of the static layer but with legend removed for rendering
        static_layer_copy = LayerDefinition(
            layer_type=static_layer.layer_type,
            geodataframe=static_layer.geodataframe,
            layer_style=static_layer.layer_style,
            legend=None,  # Remove legend from static layer
        )
        flat_static_no_legend.append(static_layer_copy)

        # Create a dummy grouped layer that only exists for legend display
        # Use the first grouped layer's properties but with the static layer's legend
        if flat_grouped:
            legend_carrier = LayerDefinition(
                layer_type=flat_grouped[0].layer_type,
                geodataframe=flat_grouped[0].geodataframe,
                layer_style=flat_grouped[0].layer_style,
                legend=static_layer.legend,  # Transfer the legend here
            )
            grouped_with_static_legends.append(legend_carrier)

    # Combine: static_no_legend (bottom) -> all grouped layers with transferred legends (top)
    all_layers = flat_static_no_legend + flat_grouped + grouped_with_static_legends

    # Separate text layers (always render on top)
    text_layers = []
    other_layers = []
    for layer in all_layers:
        if isinstance(layer.layer_style, TextLayerStyle):
            text_layers.append(layer)
        else:
            other_layers.append(layer)

    return other_layers + text_layers


@task
def order_deckgl_layers_by_type(
    layers: Annotated[List[LayerDefinition], Field(description="List of layer definitions to order by type priority.")],
) -> List[LayerDefinition]:
    """
    Order layers by type priority:
    1. TextLayerStyle (top - rendered last, appears on top)
    2. PathLayerStyle
    3. GeoJSONLayerStyle
    4. PolygonLayerStyle
    5. ScatterplotLayerStyle
    6. IconLayerStyle
    7. Other layers

    Args:
        layers: List of LayerDefinition objects to sort.

    Returns:
        Sorted list of LayerDefinition objects.
    """
    priority_map = {
        PolygonLayerStyle: 1,
        GeoJSONLayerStyle: 2,
        PathLayerStyle: 3,
        ScatterplotLayerStyle: 4,
        IconLayerStyle: 5,
        TextLayerStyle: 6,
    }

    def get_priority(layer: LayerDefinition) -> int:
        """Get sort priority for a layer based on its style type."""
        style_type = type(layer.layer_style)
        return priority_map.get(style_type, 0)

    sorted_layers = sorted(layers, key=get_priority)
    layer_types = [type(layer.layer_style).__name__ for layer in sorted_layers]
    logger.info(f"Ordered {len(sorted_layers)} layers: {layer_types}")
    return sorted_layers


@task
def get_gdf_geom_type(gdf: AnyGeoDataFrame) -> GdfWithGeometryType:
    if gdf is None or gdf.empty:
        raise ValueError("Geodataframe is empty")

    if gdf.geometry.isna().all():
        raise ValueError("Geodataframe has no valid geometries")

    unique_types = gdf.geometry.geom_type.dropna().unique().tolist()
    geom = unique_types[0]
    mapping: Dict[str, Literal["Polygon", "Point", "LineString", "Other"]] = {
        "Polygon": "Polygon",
        "MultiPolygon": "Polygon",
        "Point": "Point",
        "MultiPoint": "Point",
        "LineString": "LineString",
        "MultiLineString": "LineString",
    }
    primary_type = mapping.get(geom, "Other")
    return {"gdf": gdf, "geometry_type": primary_type}


@task
def annotate_gdf_dict_with_geom_type(
    gdf_dict: Dict[str, AnyGeoDataFrame],
) -> Dict[str, GdfWithGeometryType]:
    """
    Annotates each GeoDataFrame in the dictionary with its primary geometry type.
    Args:
        gdf_dict: Dictionary of GeoDataFrames.
    Returns:
        Dict with keys preserved, each value containing the GeoDataFrame and its geometry type.
    """
    return {name: get_gdf_geom_type(gdf) for name, gdf in gdf_dict.items()}


# upstream this
@task
def create_custom_text_layer(
    geodataframe: Annotated[
        AnyGeoDataFrame,
        Field(description="The geodataframe to visualize.", exclude=True),
    ],
    layer_style: Annotated[
        TextLayerStyle | SkipJsonSchema[None],
        AdvancedField(default=TextLayerStyle(), description="Style arguments for the layer."),
    ] = None,
    legend: Annotated[
        LegendDefinition | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="If present, includes this layer in the map legend",
        ),
    ] = None,
    use_centroid: bool = True,
) -> Annotated[LayerDefinition, Field()]:
    """
    Creates a text layer definition based on the provided configuration.
    """
    gdf = geodataframe.copy()

    if use_centroid:
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid]
        gdf = gdf.to_crs("EPSG:3857")
        gdf["geometry"] = gdf.geometry.centroid
        gdf = gdf.to_crs("EPSG:4326")
    return LayerDefinition(
        layer_type="TextLayer",
        geodataframe=gdf,
        layer_style=layer_style or TextLayerStyle(),
        legend=legend,
    )


@task
def create_deckgl_layer_from_gdf(
    gdf: GdfWithGeometryType, style: DeckGLStyleParams, legend: LegendDefinition
) -> Optional[LayerDefinition]:
    """
    Creates an appropriate layer based on GeoDataFrame geometry type.
    Mixed and Other geometry types default to Polygon layer.

    Args:
        gdf: GeoDataFrame with geometry type annotation
        style: Style parameters (will be converted to appropriate style class)
        legend: Legend definition for the layer

    Returns:
        Created layer or None if error occurs
    """
    primary = gdf["geometry_type"]
    gdf_data = gdf["gdf"]

    layer_creators = {
        "Polygon": (create_geojson_layer, GeoJSONLayerStyle),
        "Point": (create_scatterplot_layer, ScatterplotLayerStyle),
        "LineString": (create_path_layer, PathLayerStyle),
        "Mixed": (create_geojson_layer, GeoJSONLayerStyle),
        "Other": (create_geojson_layer, GeoJSONLayerStyle),
    }

    if primary not in layer_creators:
        logger.warning("Unknown geometry type '%s', defaulting to Polygon", primary)
        creator_func, style_class = create_geojson_layer, GeoJSONLayerStyle
    else:
        creator_func, style_class = layer_creators[primary]

    try:
        layer_style = style_class(**style)

        logger.info(
            "Creating %s layer (geometry type: %s)",
            style_class.__name__.replace("LayerStyle", "").lower(),
            primary,
        )
        return creator_func(gdf_data, layer_style=layer_style, legend=legend)

    except TypeError as e:
        logger.error("Error creating layer for geometry type '%s': %s", primary, e, exc_info=True)
        return None


@task
def create_deckgl_layers_from_gdf_dict(
    gdf_dict: Dict[str, GdfWithGeometryType],
    styles: Dict[str, DeckGLStyleParams] | DeckGLStyleParams,
    legends: Dict[str, LegendDefinition] | LegendDefinition,
) -> List[LayerDefinition]:
    """
    Creates styled map layers from a dictionary of GeoDataFrames with geometry types.
    Args:
        gdf_dict: Dictionary where keys are layer names and values are GeoDataFrames with geometry type
        styles: Either a single StyleParams to apply to all, or dict mapping layer names to styles
        legends: Either a single LegendDefinition to apply to all, or dict mapping layer names to legends
    Returns:
        List of LayerDefinition objects for mapping (excludes failed layers)
    """
    layers: List[LayerDefinition] = []
    use_single_style = not isinstance(styles, dict)
    use_single_legend = not isinstance(legends, dict)
    is_first_layer = True

    for name, gdf_with_geom in gdf_dict.items():
        logger.info("Processing layer '%s'", name)
        try:
            layer_style = styles if use_single_style else styles.get(name)
            layer_legend = legends if use_single_legend else legends.get(name)

            # Only use legend for the first layer if using single legend
            if use_single_legend and not is_first_layer:
                layer_legend = None

            if layer_style is None:
                logger.warning("No style provided for layer '%s', skipping", name)
                continue

            if layer_legend is None and (not use_single_legend or is_first_layer):
                logger.warning("No legend provided for layer '%s', skipping", name)
                continue

            layer = create_deckgl_layer_from_gdf(gdf=gdf_with_geom, style=layer_style, legend=layer_legend)

            if layer:
                layers.append(layer)
                logger.info("Successfully created layer for '%s'", name)
                is_first_layer = False
            else:
                logger.warning("Failed to create layer for '%s'", name)

        except Exception as e:
            logger.error("Error processing layer '%s': %s", name, e, exc_info=True)

    logger.info("Created %d layers from gdf_dict", len(layers))
    return layers


@task
def envelope_gdf(
    gdf: Annotated[
        AnyGeoDataFrame,
        Field(description="Input GeoDataFrame to create envelope from"),
    ],
    expansion_factor: Annotated[
        float,
        Field(description="Factor to expand the bounding box (e.g., 1.2 = 20% larger)"),
    ] = 1.05,
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
    if expansion_factor <= 0:
        raise ValueError("expansion_factor must be greater than 0")

    # Get the envelope (bounding box) of all geometries combined
    envelope = gdf.union_all().envelope

    # Get the bounds of the envelope
    minx, miny, maxx, maxy = envelope.bounds

    # Calculate center point
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    # Calculate current width and height
    width = maxx - minx
    height = maxy - miny

    # Apply expansion factor
    new_width = width * expansion_factor
    new_height = height * expansion_factor

    # Calculate new bounds centered on the original center
    new_minx = center_x - new_width / 2
    new_maxx = center_x + new_width / 2
    new_miny = center_y - new_height / 2
    new_maxy = center_y + new_height / 2

    # Create expanded bounding box
    expanded_envelope = box(new_minx, new_miny, new_maxx, new_maxy)

    # Convert to GeoDataFrame with same CRS as input
    envelope_gdf = gpd.GeoDataFrame({"geometry": [expanded_envelope]}, crs=gdf.crs)

    return envelope_gdf
