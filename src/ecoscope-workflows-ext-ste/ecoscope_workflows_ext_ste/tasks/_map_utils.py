import os
import re
import math
import logging
import traceback
from enum import Enum
from pydantic import BaseModel, Field
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyGeoDataFrame
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import ViewState
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import TextLayerStyle
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import PointLayerStyle
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import LayerDefinition
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import LegendDefinition
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import PolygonLayerStyle
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import PolylineLayerStyle
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import create_point_layer
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import create_polygon_layer
from ecoscope_workflows_ext_ecoscope.tasks.results._ecomap import create_polyline_layer
from typing import Union,Any, Dict, Optional, Literal, List, Annotated, TypedDict, Tuple

logger = logging.getLogger(__name__)

class MapStyleConfig(BaseModel):
    styles: Dict[str, Dict] = Field(default_factory=dict)
    legend: Dict[str, List[str]] = Field(default_factory=dict)

class SupportedFormat(str, Enum):
    GPKG = ".gpkg"
    GEOJSON = ".geojson"
    SHP = ".shp"

SUPPORTED_FORMATS = [f.value for f in SupportedFormat]

class MapProcessingConfig(BaseModel):
    path: str = Field(..., description="Directory path to load geospatial files from")
    target_crs: Union[int, str] = Field(default=4326, description="Target CRS to convert maps to")
    recursive: bool = Field(default=False, description="Whether to walk folders recursively")


class GeometrySummary(TypedDict):
    primary_type: Literal["Polygon", "Point", "LineString", "Other", "Mixed", "Line"]


@task
def remove_invalid_geometries(
    gdf: Annotated[AnyGeoDataFrame, Field(description="GeoDataFrame to filter for valid geometries.", exclude=True)],
) -> AnyGeoDataFrame:
    """
    Remove rows from the GeoDataFrame that contain null or empty geometries.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame that may contain invalid geometries.

    Returns:
        GeoDataFrame: Filtered GeoDataFrame containing only non-empty, non-null geometries.
    """
    return gdf.loc[(~gdf.geometry.isna()) & (~gdf.geometry.is_empty)]

# upstream 
@task
def detect_geometry_type(gdf: AnyGeoDataFrame) -> GeometrySummary:
    """
    Detect the dominant geometry type in a GeoDataFrame and count each type.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame whose geometries will be analyzed.

    Returns:
        GeometrySummary: Dict containing the detected geometry type and counts per geometry class.
    """
    geom_counts = gdf.geometry.geom_type.value_counts().to_dict()
    unique_types = list(geom_counts.keys())

    if len(unique_types) == 1:
        geom = unique_types[0]
        mapping = {
            "Polygon": "Polygon",
            "MultiPolygon": "Polygon",
            "Point": "Point",
            "MultiPoint": "Point",
            "LineString": "LineString",
            "MultiLineString": "LineString",
        }
        primary_type = mapping.get(geom, "Other")
    else:
        primary_type = "Mixed"

    return {"primary_type": primary_type, "counts": geom_counts}

def clean_file_keys(file_dict: Dict[str, AnyGeoDataFrame]) -> Dict[str, AnyGeoDataFrame]:
    """
    Clean dictionary keys by removing file extensions and normalizing names.
    Args:
        file_dict: Dictionary mapping filenames to GeoDataFrames.
    Returns:
        A new dictionary with standardized, lowercase keys suitable for map layer identifiers.
    """
    def clean_key(key: str) -> str:
        for ext in SUPPORTED_FORMATS:
            if key.lower().endswith(ext):
                key = key[: -len(ext)]
                break

        key = re.sub(r'\band\b', '_', key, flags=re.IGNORECASE)
        key = re.sub(r'[^A-Za-z0-9_]+', '_', key)
        key = re.sub(r'_+', '_', key)  # collapse multiple underscores
        return key.strip('_').lower()
    return {clean_key(k): v for k, v in file_dict.items()}

# upstream 
@task
def create_layer_from_gdf(
    filename: str,
    gdf: AnyGeoDataFrame,
    style_config: MapStyleConfig,
) -> Optional[LayerDefinition]:
    """
    Create an appropriately styled map layer from a GeoDataFrame.

    Args:
        filename: key matching an entry in style_config.styles (use cleaned keys).
        gdf: GeoDataFrame to render.
        style_config: MapStyleConfig containing style definitions and optional legend.

    Returns:
        A LayerDefinition (or subclass) or None if creation failed / unsupported.
    """
    if filename not in style_config.styles:
        logger.warning("No style config for '%s'", filename)
        return None

    style_params: Dict[str, Any] = style_config.styles[filename]

    legend = None
    legend_cfg = getattr(style_config, "legend", None)
    if isinstance(legend_cfg, dict):
        labels = legend_cfg.get("labels")
        colors = legend_cfg.get("colors")
        if labels and colors and len(labels) == len(colors):
            legend = LegendDefinition(labels=labels, colors=colors)
        else:
            logger.debug("Skipping legend for '%s' due to missing or mismatched labels/colors", filename)
    
    gdf = gdf.loc[(~gdf.geometry.isna()) & (~gdf.geometry.is_empty)]
    primary = detect_geometry_type(gdf=gdf)["primary_type"]

    try:
        if primary == "Polygon":
            logger.info("Creating polygon layer for '%s'", filename)
            return create_polygon_layer(gdf, layer_style=PolygonLayerStyle(**style_params), legend=legend)

        if primary == "Point":
            logger.info("Creating point layer for '%s'", filename)
            return create_point_layer(gdf, layer_style=PointLayerStyle(**style_params), legend=legend)

        if primary == "LineString":
            logger.info("Creating line layer for '%s'", filename)
            return create_polyline_layer(gdf, layer_style=PolylineLayerStyle(**style_params), legend=legend)
        
    except TypeError as te:
        logger.error("Invalid style params for '%s': %s", filename, te, exc_info=True)
    return None

# upstream 
@task
def create_map_layers(
    file_dict: Dict[str, AnyGeoDataFrame], 
    style_config: MapStyleConfig
    ) -> List[LayerDefinition]:
    """
    Create styled map layers from a dictionary of GeoDataFrames using the provided style config.

    Args:
        file_dict: Dictionary mapping filenames to GeoDataFrames.
        style_config: Object holding style definitions and legend config.
    Returns:
        A list of styled map layer objects.
    """
    layers: List[LayerDefinition] = []
    cleaned_files = clean_file_keys(file_dict)

    for filename, gdf in cleaned_files.items():
        try:
            gdf = gdf.loc[(~gdf.geometry.isna()) & (~gdf.geometry.is_empty)]
            layer = create_layer_from_gdf(filename, gdf, style_config)

            if layer is not None:
                layers.append(layer)

        except Exception as e:
            logger.error("Error processing layer for '%s': %s", filename, e, exc_info=True)
    logger.info("Successfully created %d map layers", len(layers))
    return layers


# https://stackoverflow.com/questions/63787612/plotly-automatic-zooming-for-mapbox-maps
# Alternative approach using both dimensions for better fitting
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

# upstream 
@task
def create_view_state_from_gdf(
    gdf: AnyGeoDataFrame, 
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
    return ViewState(
        longitude=center_lon, 
        latitude=center_lat, 
        zoom=zoom, 
        pitch=pitch, 
        bearing=bearing
        )

# landDx style config specific -- upstream?
@task
def build_landdx_style_config(aoi_list: List[str], color_map: Dict[str, Tuple[int, int, int]]) -> MapStyleConfig:
    """
    Build a MapStyleConfig object with styles and legends for specified AOI types.

    Args:
        aoi_list (List[str]): List of area of interest (AOI) types.
        color_map (Dict[str, Tuple[int, int, int]]): Mapping from AOI type to RGB color.

    Returns:
        MapStyleConfig: A style configuration with defined styles and legends.
    """
    styles: Dict[str, Dict[str, object]] = {}
    legend: Dict[str, List[str]] = {"labels": [], "colors": []}

    for aoi in aoi_list:
        rgb = color_map.get(aoi, (128, 128, 128))  # fallback to gray
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)

        styles[aoi] = {
            "get_elevation": 1000,
            "get_fill_color": rgb,
            "opacity": 0.25,
            "get_line_width": 4.00,
            "get_width": 5,
            "get_line_color": rgb,
            "get_line_width": 2,
            "stroked":True
        }

        legend["labels"].append(aoi)
        legend["colors"].append(hex_color)
    return MapStyleConfig(styles=styles, legend=legend)

# upstream 
@task
def annotate_gdf_dict_with_geometry_type(gdf_dict: Dict[str, AnyGeoDataFrame]) -> Dict[str, Dict[str, object]]:
    """
    Annotates each GeoDataFrame in the dictionary with its primary geometry type.

    Args:
        gdf_dict: Dictionary of GeoDataFrames.

    Returns:
        Dict with keys preserved, each value containing the GeoDataFrame and its geometry type.
    """
    result = {}
    
    for name, gdf in gdf_dict.items():
        geometry_summary = detect_geometry_type(gdf)
        result[name] = {
            "gdf": gdf, 
            "geometry_type": geometry_summary["primary_type"]
        }

    return result

#upstream
@task
def create_map_layers_from_annotated_dict(
    annotated_dict: Dict[str, Dict[str, object]], 
    style_config: MapStyleConfig
) -> List[LayerDefinition]:
    """
    Create styled map layers from a dictionary of {name: {gdf, geometry_type}}.

    Args:
        annotated_dict: A dictionary where each value contains a GeoDataFrame and its geometry type.
        style_config: A MapStyleConfig object that defines styles for each name key.

    Returns:
        List of LayerDefinition objects for mapping.
    """
    # Convert dict to MapStyleConfig if needed
    if isinstance(style_config, dict):
        style_config = MapStyleConfig(**style_config)

    layers: List[LayerDefinition] = []

    for name, content in annotated_dict.items():
        gdf = content.get("gdf")

        try:
            layer = create_layer_from_gdf(
                filename=name, 
                gdf=gdf, 
                style_config=style_config, 
                )

            if layer:
                layers.append(layer)
        except Exception as e:
            logger.error(f"Error creating layer for '{name}': {e}")
            traceback.print_exc()

    logger.info(f"Created {len(layers)} layers from annotated dict")
    return layers

# upstream 
@task
def combine_map_layers(
    static_layers: Annotated[
        Union[LayerDefinition, List[LayerDefinition | List[LayerDefinition]]], 
        Field(description="Static layers from local files or base maps.")
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
    txt_gdf: Annotated[
        AnyGeoDataFrame,
        Field(description="Input GeoDataFrame containing geometries and label data.")
    ],
    label_column: Annotated[
        str,
        Field(default="label", description="Column name containing text labels.")
    ] = "label",
    name_column: Annotated[
        str,
        Field(default="name", description="Fallback column name to use as label if label_column doesn’t exist.")
    ] = "name",
    use_centroid: Annotated[
        bool,
        Field(default=True, description="Whether to use geometry centroids for text placement.")
    ] = True,
    color: Annotated[
        List[int],
        Field(default=[0, 0, 0, 255], description="RGBA color values for text (0–255).")
    ] = [0, 0, 0, 255],
    size: Annotated[
        int,
        Field(default=16, description="Font size in pixels.")
    ] = 16,
    font_weight: Annotated[
        str,
        Field(default="normal", description="Font weight (e.g., normal, bold, italic).")
    ] = "normal",
    font_family: Annotated[
        str,
        Field(default="Arial", description="Font family name.")
    ] = "Arial",
    text_anchor: Annotated[
        str,
        Field(default="middle", description="Horizontal text anchor (start, middle, end).")
    ] = "middle",
    alignment_baseline: Annotated[
        str,
        Field(default="center", description="Vertical alignment (top, center, bottom).")
    ] = "center",
    pickable: Annotated[
        bool,
        Field(default=True, description="Whether the layer is interactive (pickable).")
    ] = True,
    tooltip_columns: Annotated[
        Optional[List[str]],
        Field(default=None, description="Columns to display in tooltip when hovered.")
    ] = None,
    zoom: Annotated[
        bool,
        Field(default=False, description="Whether to zoom to the layer extent when displayed.")
    ] = False,
    target_crs: Annotated[
        str,
        Field(default="epsg:4326", description="Target CRS for layer coordinates.")
    ] = "epsg:4326"
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
                f"Neither '{label_column}' nor '{name_column}' found. "
                f"Available columns: {list(gdf.columns)}"
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

# upstream?
@task
def find_landdx_gpkg_path(
    output_dir: str,
) -> Optional[str]:
    """
    Search for landDx.gpkg file in the output directory and return its full path.
    
    Args:
        output_dir: Directory path to search for landDx.gpkg
    
    Returns:
        Full path to landDx.gpkg file if found, None otherwise
    """
    if output_dir is None or output_dir == "": 
        logger.error("Provided output directory is empty")
        return None
    
    if not os.path.exists(output_dir):
        logger.error(f"Output directory does not exist: {output_dir}")
        return None
    
    for root, _, files in os.walk(output_dir):
        if "landDx.gpkg" in files:
            ldx_path = os.path.join(root, "landDx.gpkg")
            logger.info(f"Found landDx.gpkg at: {ldx_path}")
            return ldx_path
    
    logger.error(f"landDx.gpkg not found in {output_dir}")
    return None