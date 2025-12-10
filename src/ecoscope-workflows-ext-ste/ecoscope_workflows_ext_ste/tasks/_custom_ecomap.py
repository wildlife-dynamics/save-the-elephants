import math
import logging
from enum import Enum
from pydantic import Field
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyGeoDataFrame
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict, Union
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import (
    LayerDefinition,
    LegendDefinition,
    PointLayerStyle,
    PolygonLayerStyle,
    PolylineLayerStyle,
    TextLayerStyle,
    ViewState,
    create_point_layer,
    create_polygon_layer,
    create_polyline_layer,
)

logger = logging.getLogger(__name__)


class SupportedFormat(str, Enum):
    GPKG = ".gpkg"
    GEOJSON = ".geojson"
    SHP = ".shp"


class GdfWithGeometryType(TypedDict):
    gdf: AnyGeoDataFrame
    geometry_type: str


class StyleParams(TypedDict, total=False):
    """Universal style parameters that can be used for any layer type."""

    # Base style parameters (common to all)
    auto_highlight: bool
    opacity: float
    pickable: bool

    # Shape style parameters (Polygon and Point)
    filled: bool
    get_fill_color: str | list[int] | list[list[int]] | None
    get_line_color: str | list[int] | list[list[int]] | None
    get_line_width: float
    fill_color_column: str | None
    line_width_units: Literal["pixels", "meters"]
    stroked: bool

    # Polygon specific
    extruded: bool
    get_elevation: float

    # Point specific
    get_radius: float
    radius_units: Literal["pixels", "meters"]

    # Polyline specific
    get_color: str | list[int] | list[list[int]] | None
    get_width: float
    color_column: str | None
    width_units: Literal["pixels", "meters"]
    cap_rounded: bool


SUPPORTED_FORMATS = [f.value for f in SupportedFormat]


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
def create_view_state_from_gdf(
    gdf: AnyGeoDataFrame,
    pitch: int = 0,
    bearing: int = 0,
) -> ViewState:
    """Create a ViewState centered on the GeoDataFrame's bounding box."""
    if gdf.empty:
        raise ValueError("GeoDataFrame is empty. Cannot compute ViewState.")
    minx, miny, maxx, maxy = gdf.total_bounds
    center_lon = (minx + maxx) / 2.0
    center_lat = (miny + maxy) / 2.0
    zoom = _zoom_from_bbox(minx, miny, maxx, maxy)
    return ViewState(
        longitude=center_lon,
        latitude=center_lat,
        zoom=zoom,
        pitch=pitch,
        bearing=bearing,
    )


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


@task
def create_layer_from_gdf(gdf: GdfWithGeometryType, style: StyleParams, legend: LegendDefinition) -> Optional[Any]:
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

    # Mapping of geometry types to layer creation functions and style classes
    layer_creators = {
        "Polygon": (create_polygon_layer, PolygonLayerStyle),
        "Point": (create_point_layer, PointLayerStyle),
        "LineString": (create_polyline_layer, PolylineLayerStyle),
        "Mixed": (create_polygon_layer, PolygonLayerStyle),
        "Other": (create_polygon_layer, PolygonLayerStyle),
    }

    if primary not in layer_creators:
        logger.warning("Unknown geometry type '%s', defaulting to Polygon", primary)
        creator_func, style_class = create_polygon_layer, PolygonLayerStyle
    else:
        creator_func, style_class = layer_creators[primary]

    try:
        # Instantiate the appropriate style class from the dictionary
        layer_style = style_class(**style)

        logger.info(
            "Creating %s layer (geometry type: %s)", style_class.__name__.replace("LayerStyle", "").lower(), primary
        )
        return creator_func(gdf_data, layer_style=layer_style, legend=legend)

    except TypeError as e:
        logger.error("Error creating layer for geometry type '%s': %s", primary, e, exc_info=True)
        return None


@task
def create_layers_from_gdf_dict(
    gdf_dict: Dict[str, GdfWithGeometryType],
    styles: Dict[str, StyleParams] | StyleParams,
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

    # Determine if we're using single or per-layer styles/legends
    use_single_style = not isinstance(styles, dict)
    use_single_legend = not isinstance(legends, dict)

    for name, gdf_with_geom in gdf_dict.items():
        logger.info("Processing layer '%s'", name)

        try:
            # Get the appropriate style and legend for this layer
            layer_style = styles if use_single_style else styles.get(name)
            layer_legend = legends if use_single_legend else legends.get(name)

            if layer_style is None:
                logger.warning("No style provided for layer '%s', skipping", name)
                continue

            if layer_legend is None:
                logger.warning("No legend provided for layer '%s', skipping", name)
                continue

            layer = create_layer_from_gdf(gdf=gdf_with_geom, style=layer_style, legend=layer_legend)

            if layer:
                layers.append(layer)
                logger.info("Successfully created layer for '%s'", name)
            else:
                logger.warning("Failed to create layer for '%s'", name)

        except Exception as e:
            logger.error("Error processing layer '%s': %s", name, e, exc_info=True)

    logger.info("Created %d layers from gdf_dict", len(layers))
    return layers


@task
def combine_map_layers(
    static_layers: Annotated[
        Union[LayerDefinition, List[LayerDefinition | List[LayerDefinition]]],
        Field(description="Static layers from local files or base maps."),
    ] = [],
    grouped_layers: Annotated[
        Union[LayerDefinition, List[LayerDefinition | List[LayerDefinition]]],
        Field(description="Grouped layers generated from split/grouped data."),
    ] = [],
) -> list[LayerDefinition]:
    """
    Combine static and grouped map layers into a single list for rendering in `draw_ecomap`.
    Automatically flattens nested lists to handle cases where layer generation tasks return lists.
    """

    def flatten_layers(layers):
        """Recursively flatten nested lists of LayerDefinition objects."""
        if not isinstance(layers, list):
            return [layers]

        flattened = []
        for item in layers:
            if isinstance(item, list):
                # Recursively flatten if it's a list
                flattened.extend(flatten_layers(item))
            else:
                # Add individual LayerDefinition objects
                flattened.append(item)
        return flattened

    # Flatten both static and grouped layers
    flat_static = flatten_layers(static_layers) if static_layers else []
    flat_grouped = flatten_layers(grouped_layers) if grouped_layers else []

    # Combine all layers
    all_layers = flat_static + flat_grouped

    # Separate text layers from other layers
    text_layers = []
    other_layers = []

    for layer in all_layers:
        if isinstance(layer.layer_style, TextLayerStyle):
            text_layers.append(layer)
        else:
            other_layers.append(layer)

    return other_layers + text_layers


@task
def make_text_layer(
    txt_gdf: Annotated[AnyGeoDataFrame, Field(description="Input GeoDataFrame containing geometries and label data.")],
    label_column: Annotated[str, Field(default="label", description="Column name containing text labels.")] = "label",
    name_column: Annotated[
        str, Field(default="name", description="Fallback column name to use as label if label_column doesn’t exist.")
    ] = "name",
    use_centroid: Annotated[
        bool, Field(default=True, description="Whether to use geometry centroids for text placement.")
    ] = True,
    color: Annotated[List[int], Field(default=[0, 0, 0, 255], description="RGBA color values for text (0–255).")] = [
        0,
        0,
        0,
        255,
    ],
    size: Annotated[int, Field(default=16, description="Font size in pixels.")] = 16,
    font_weight: Annotated[
        str, Field(default="normal", description="Font weight (e.g., normal, bold, italic).")
    ] = "normal",
    font_family: Annotated[str, Field(default="Arial", description="Font family name.")] = "Arial",
    text_anchor: Annotated[
        str, Field(default="middle", description="Horizontal text anchor (start, middle, end).")
    ] = "middle",
    alignment_baseline: Annotated[
        str, Field(default="center", description="Vertical alignment (top, center, bottom).")
    ] = "center",
    pickable: Annotated[bool, Field(default=True, description="Whether the layer is interactive (pickable).")] = True,
    tooltip_columns: Annotated[
        Optional[List[str]], Field(default=None, description="Columns to display in tooltip when hovered.")
    ] = None,
    zoom: Annotated[
        bool, Field(default=False, description="Whether to zoom to the layer extent when displayed.")
    ] = False,
    target_crs: Annotated[
        str, Field(default="epsg:4326", description="Target CRS for layer coordinates.")
    ] = "epsg:4326",
) -> LayerDefinition:
    """
    Create a text layer from a GeoDataFrame with annotated parameters.
    """
    # Validate input
    if txt_gdf is None or txt_gdf.empty:
        raise ValueError("txt_gdf cannot be None or empty")

    gdf = txt_gdf.copy()

    # Handle label column
    if label_column not in gdf.columns:
        if name_column in gdf.columns:
            gdf = gdf.rename(columns={name_column: label_column})
        else:
            raise ValueError(
                f"Neither '{label_column}' nor '{name_column}' found. " f"Available columns: {list(gdf.columns)}"
            )

    # Use centroids if requested
    if use_centroid:
        gdf["geometry"] = gdf.centroid

    # Transform to target CRS
    gdf = gdf.to_crs(target_crs)

    # Build text style
    style = TextLayerStyle(
        get_color=color,
        get_size=size,
        font_weight=font_weight,
        get_text_anchor=text_anchor,
        get_alignment_baseline=alignment_baseline,
        font_family=font_family,
        pickable=pickable,
    )

    # Return the layer definition
    return LayerDefinition(
        geodataframe=gdf,
        layer_style=style,
        legend=None,
        tooltip_columns=tooltip_columns,
        zoom=zoom,
    )
