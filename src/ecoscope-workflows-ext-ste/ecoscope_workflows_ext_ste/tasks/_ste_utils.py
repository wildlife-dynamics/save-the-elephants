from __future__ import annotations
import os
import uuid
import logging
import hashlib
import zipfile
import numpy as np 
import pandas as pd
import geopandas as gpd
from pathlib import Path
from docx.shared import Cm
from datetime import datetime
from urllib.parse import urlparse
from ecoscope.io import download_file
from urllib.request import url2pathname
from dataclasses import asdict,dataclass
from ecoscope.trajectory import Trajectory
from ecoscope.base.utils import hex_to_rgba
from docxtpl import DocxTemplate,InlineImage 
from pydantic.json_schema import SkipJsonSchema
from dateutil.relativedelta import relativedelta
from pydantic import Field, BaseModel, ConfigDict
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.indexes import CompositeFilter
from ecoscope.analysis.ecograph import Ecograph, get_feature_gdf
from typing import Annotated, Optional, Dict, cast, Literal,Union
from ecoscope_workflows_core.tasks.filter._filter import TimeRange 
from ecoscope.analysis.seasons import seasonal_windows, std_ndvi_vals, val_cuts
from ecoscope_workflows_core.skip import SkippedDependencyFallback, SkipSentinel
from ecoscope_workflows_ext_ecoscope.tasks.analysis import calculate_elliptical_time_density
from ecoscope_workflows_core.annotations import AnyGeoDataFrame, AnyDataFrame, AdvancedField

logger = logging.getLogger(__name__)

@dataclass
class MapbookContext:
    subject_name: Optional[str] = None
    time_period: Optional[str] = None
    period: Optional[Union[int, float]] = None
    grid_area: Optional[Union[int, float]] = None
    mcp_area: Optional[Union[int, float]] = None
    movement_tracks_ecomap: Optional[str] = None
    home_range_ecomap: Optional[str] = None
    speedmap: Optional[str] = None
    speed_raster_ecomap: Optional[str] = None
    night_day_ecomap: Optional[str] = None
    seasonal_homerange: Optional[str] = None

class AutoScaleGridCellSize(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "Auto-scale"})
    auto_scale_or_custom: Annotated[
        Literal["Auto-scale"],
        AdvancedField(
            default="Auto-scale",
            title=" ",
            description="Define the resolution of the raster grid (in meters per pixel).",
        ),
    ] = "Auto-scale"

class CustomGridCellSize(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "Customize"})
    auto_scale_or_custom: Annotated[
        Literal["Customize"],
        AdvancedField(
            default="Customize",
            title=" ",
            description="Define the resolution of the raster grid (in meters per pixel).",
        ),
    ] = "Customize"
    grid_cell_size: Annotated[
        float | SkipJsonSchema[None],
        Field(
            description="Custom Raster Pixel Size (Meters)",
            gt=0,
            lt=10000,
            default=5000,
            json_schema_extra={"exclusiveMinimum": 0, "exclusiveMaximum": 10000},
        ),
    ] = 5000

@task
def label_quarter_status(gdf: AnyDataFrame, timestamp_col: str) -> AnyDataFrame:
    """
    Label rows based on whether their timestamp falls in the most recent quarter.

    Args:
        gdf: Input DataFrame with a datetime column.
        timestamp_col: Name of the timestamp column.

    Returns:
        A copy of the DataFrame with a new column 'quarter_status' indicating
        'Present Quarter Movement' or 'Previous Quarter Movement'.
    """
    if gdf is None or gdf.empty:
        raise ValueError("`label_quarter_status`:gdf is empty.")
    
    gdf[timestamp_col] = pd.to_datetime(gdf[timestamp_col])
    latest_date = gdf[timestamp_col].max()
    most_recent_quarter = latest_date.to_period("Q")
    gdf["quarter_status"] = np.where(
        gdf[timestamp_col].dt.to_period("Q") == most_recent_quarter,
        "Present Quarter Movement",
        "Previous Quarter Movement",
    )
    return gdf

@task
def generate_ecograph_raster(
    gdf: Annotated[AnyGeoDataFrame, Field(description="GeoDataFrame with trajectory data")],
    dist_col: Annotated[str, Field(description="Column name for step distance")],
    output_dir: Annotated[Optional[str], Field(description="Directory to save the output raster")],
    filename: Annotated[
        Optional[str],
        Field(
            description="Filename for the output GeoTIFF (without extension). "
            "If not provided, a hash of the data will be used.",
            exclude=True,
        ),
    ] = None,
    resolution: Annotated[
        Optional[float], Field(default=None, description="Raster resolution; if None, uses the mean of dist_col.")
    ] = None,
    radius: Annotated[int, Field(default=2, description="Radius for kernel smoothing")] = 2,
    cutoff: Annotated[Optional[float], Field(default=None, description="Cutoff distance for kernel")] = None,
    tortuosity_length: Annotated[int, Field(default=3, description="Length scale for tortuosity smoothing")] = 3,
    interpolation: Literal["mean", "min", "max", "median"] = "mean",
    step_length: Annotated[Optional[int], Field(default=None, description="Mean step length for resolution")] = None,
    movement_covariate: Optional[
        Literal["dot_product", "step_length", "speed", "sin_time", "cos_time", "tortuosity_1", "tortuosity_2"]
    ] = None,
    network_metric: Optional[Literal["weight", "betweenness", "degree", "collective_influence"]] = None,
) -> str:

    if gdf is None or gdf.empty:
        raise ValueError("`generate_ecograph_raster`:Trajectory gdf is empty.")

    if dist_col not in gdf.columns:
        raise ValueError(f"`generate_ecograph_raster`:Column '{dist_col}' not found in gdf.")

    dist_series = pd.to_numeric(gdf[dist_col], errors="coerce")
    if dist_series.dropna().empty:
        raise ValueError(f"`generate_ecograph_raster`:Column '{dist_col}' has no numeric values to compute a mean resolution.")

    if (movement_covariate is None) == (network_metric is None):
        raise ValueError("`generate_ecograph_raster`:Provide exactly one of 'movement_covariate' or 'network_metric'.")
        
    if output_dir is None or str(output_dir).strip() == "":
        output_dir = os.getcwd()
    else:
        output_dir = str(output_dir).strip()

    if output_dir.startswith("file://"):
        parsed = urlparse(output_dir)
        output_dir = url2pathname(parsed.path)

    if not filename:
        df_hash = hashlib.sha256(pd.util.hash_pandas_object(gdf, index=True).values).hexdigest()
        filename = df_hash[:7]
    if step_length is not None:
        resolution = float(step_length)
    else:
        step_length = float(dist_series.mean())
    res = float(resolution) if resolution is not None else step_length
    if res <= 0:
        raise ValueError(f"Computed/Provided resolution must be > 0, got {res}.")

    os.makedirs(output_dir, exist_ok=True)
    raster_path = os.path.join(output_dir, f"{filename}.tif")
    ecograph = Ecograph(
        Trajectory(gdf),
        resolution=res,
        radius=radius,
        cutoff=cutoff,
        tortuosity_length=tortuosity_length,
    )

    covariate = movement_covariate if movement_covariate is not None else network_metric
    ecograph.to_geotiff(covariate, raster_path, interpolation=interpolation)
    return raster_path

@task
def retrieve_feature_gdf(
    file_path: Annotated[str, Field(description="Path to the saved Ecograph feature file")],
) -> AnyGeoDataFrame:
    if not isinstance(file_path, str) or not file_path:
        raise ValueError("retrieve_feature_gdf: 'file_path' must be a non-empty string.")

    if file_path.startswith("file://"):
        parsed = urlparse(file_path)
        file_path = url2pathname(parsed.path)

    gdf = get_feature_gdf(file_path)
    return gdf

@task
def create_seasonal_labels(traj: AnyGeoDataFrame, total_percentiles: AnyDataFrame) -> Optional[AnyGeoDataFrame]:
    """
    Annotates trajectory segments with seasonal labels (wet/dry) based on NDVI-derived windows.
    Applies to the entire trajectory without grouping.
    """
    try:
        if traj is None or traj.empty:
            raise ValueError("`create_seasonal_labels`:traj gdf is empty.")
        if total_percentiles is None or total_percentiles.empty:
            raise ValueError("`create_seasonal_labels `:total_percentiles df is empty.")

        seasonal_wins = total_percentiles.copy()
        traj_start = traj["segment_start"].min()
        traj_end = traj["segment_end"].max()

        seasonal_wins = seasonal_wins[
            (seasonal_wins["end"] >= traj_start) & (seasonal_wins["start"] <= traj_end)
        ].reset_index(drop=True)

        logger.info(f"Filtered seasonal windows: {len(seasonal_wins)} periods")
        logger.info(f"Seasonal Windows:\n{seasonal_wins[['start', 'end', 'season']]}")

        if seasonal_wins.empty:
            logger.error("No seasonal windows overlap with trajectory timeframe.")
            traj["season"] = None
            return traj

        season_bins = pd.IntervalIndex(data=seasonal_wins.apply(lambda x: pd.Interval(x["start"], x["end"]), axis=1))
        logger.info(f"Created {len(season_bins)} seasonal bins")

        labels = seasonal_wins["season"].values
        traj["season"] = pd.cut(traj["segment_start"], bins=season_bins, include_lowest=True).map(
            dict(zip(season_bins, labels))
        )
        null_count = traj["season"].isnull().sum()
        if null_count > 0:
            logger.warning(f"Warning: {null_count} trajectory segments couldn't be assigned to any season")

        logger.info("Seasonal labeling complete. Season distribution:")
        logger.info(traj["season"].value_counts(dropna=False))
        return traj
    except Exception as e:
        logger.error(f"Failed to apply seasonal label to trajectory: {e}")
        return None

@task
def split_gdf_by_column(
    gdf: Annotated[AnyGeoDataFrame, Field(description="The GeoDataFrame to split")],
    column: Annotated[str, Field(description="Column name to split GeoDataFrame by")],
) -> Dict[str, AnyGeoDataFrame]:
    """
    Splits a GeoDataFrame into a dictionary of GeoDataFrames based on unique values in the specified column.
    """
    if gdf is None or gdf.empty:
        raise ValueError("`split_gdf_by_column`:gdf is empty.")

    if column not in gdf.columns:
        raise ValueError(f"`split_gdf_by_column`:Column '{column}' not found in GeoDataFrame.")

    grouped = {str(k): v for k, v in gdf.groupby(column)}
    return grouped

@task
def generate_mcp_gdf(
    gdf: AnyGeoDataFrame,
    planar_crs: str = "ESRI:102022",  # Africa Albers Equal Area
) -> AnyGeoDataFrame:
    """
    Create a Minimum Convex Polygon (MCP) from input point geometries and compute its area.
    """
    if gdf is None or gdf.empty:
        raise ValueError("`generate_mcp_gdf`:gdf is empty.")
    if gdf.geometry is None:
        raise ValueError("`generate_mcp_gdf`:gdf has no 'geometry' column.")
    if gdf.crs is None:
        raise ValueError("`generate_mcp_gdf`:gdf must have a CRS set (e.g., EPSG:4326).")

    original_crs = gdf.crs
    valid_points_gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    if valid_points_gdf.empty:
        raise ValueError("`generate_mcp_gdf`:No valid geometries in gdf.")

    if not all(valid_points_gdf.geometry.geom_type.isin(["Point"])):
        valid_points_gdf.geometry = valid_points_gdf.geometry.centroid

    projected_gdf = valid_points_gdf.to_crs(planar_crs)
    convex_hull = projected_gdf.geometry.unary_union.convex_hull

    area_sq_meters = float(convex_hull.area)
    area_sq_km = area_sq_meters / 1_000_000.0
    convex_hull_original_crs = gpd.GeoSeries([convex_hull], crs=planar_crs).to_crs(original_crs).iloc[0]

    result_gdf = gpd.GeoDataFrame(
        {
            "area_m2": [area_sq_meters], 
            "area_km2": [area_sq_km], 
            "mcp": "mcp"
        },
        geometry=[convex_hull_original_crs],
        crs=original_crs,
    )
    return result_gdf

@task
def dataframe_column_first_unique_str(
    df: AnyDataFrame,
    column_name: Annotated[str, Field(description="Column to aggregate")],
) -> Annotated[str, Field(description="The first unique string value in the column")]:
    if gdf is None or gdf.empty:
        raise ValueError("`dataframe_column_first_unique_str`:df is empty.")
    return str(df[column_name].unique()[0])


@task
def assign_quarter_status_colors(
    gdf: AnyDataFrame, 
    hex_column: str, 
    previous_color_hex: str
    ) -> AnyDataFrame:
    if gdf is None or gdf.empty:
        raise ValueError("`assign_quarter_status_colors`:gdf is empty.")

    df = gdf.copy()
    if not isinstance(previous_color_hex, str) or not previous_color_hex.startswith("#"):
        raise ValueError("`assign_quarter_status_colors`:Invalid hex color code for previous_color_hex.")

    if hex_column not in df.columns:
        raise ValueError(f"`assign_quarter_status_colors`:Column '{hex_column}' not found in gdf.")
    
    if "quarter_status" not in df.columns:
        raise ValueError("`assign_quarter_status_colors`:Column 'quarter_status' not found in gdf.")

    prev_rgba = hex_to_rgba(previous_color_hex)
    df["quarter_status_hex_colors"] = np.where(
        df["quarter_status"] == "Present Quarter Movement",
        df[hex_column],
        previous_color_hex,
    )
    df["quarter_status_colors"] = df["quarter_status_hex_colors"].apply(hex_to_rgba)
    return df

@task
def calculate_seasonal_home_range(
    gdf: AnyGeoDataFrame,
    groupby_cols: Annotated[
        list[str],
        Field(
            description="List of column names to group by (e.g., ['groupby_col', 'season'])",
            json_schema_extra={"default": ["groupby_col", "season"]},
        ),
    ] = None,
    percentiles: Annotated[
        list[float] | SkipJsonSchema[None],
        Field(default=[25.0, 50.0, 75.0, 90.0, 95.0, 99.9]),
    ] = [99.9],
    auto_scale_or_custom_cell_size: Annotated[
        AutoScaleGridCellSize | CustomGridCellSize | SkipJsonSchema[None],
        Field(
            json_schema_extra={
                "title": "Auto Scale Or Custom Grid Cell Size",
                "ecoscope:advanced": True,
                "default": {"auto_scale_or_custom": "Auto-scale"},
            },
        ),
    ] = None,
) -> AnyDataFrame:
    if gdf is None or gdf.empty:
        raise ValueError("`calculate_seasonal_home_range`:gdf is empty.")

    if groupby_cols is None:
        groupby_cols = ["groupby_col", "season"]
    
    if 'season' not in gdf.columns:
        raise ValueError("`calculate_seasonal_home_range`: gdf must have a 'season' column.")
    
    if auto_scale_or_custom_cell_size is None:
        auto_scale_or_custom_cell_size = AutoScaleGridCellSize()

    gdf = gdf[gdf['season'].notna()].copy()
    group_counts = gdf.groupby(groupby_cols).size()
    try:
        season_etd = gdf.groupby(groupby_cols).apply(
            lambda df: calculate_elliptical_time_density(
                df, 
                auto_scale_or_custom_cell_size=auto_scale_or_custom_cell_size,
                percentiles=percentiles,
            )
        )
    except TypeError:
        season_etd = gdf.groupby(groupby_cols).apply(
            lambda df: calculate_elliptical_time_density(
                df, 
                auto_scale_or_custom_cell_size=auto_scale_or_custom_cell_size,
                percentiles=percentiles,
            ),
            include_groups=False,
        )
    # Reset index properly
    if isinstance(season_etd.index, pd.MultiIndex):
        season_etd = season_etd.reset_index()
    return season_etd

@task
def get_duration(
    time_range: Annotated[
        TimeRange,
        Field(description="Time range object"),
    ],
    time_unit: Literal["days", "months"] = "months",
) -> float:
    """
    Compute the duration of a TimeRange object in days or months.

    Args:
        time_range (TimeRange): The TimeRange returned by set_time_range.
        time_unit (str): Either "days" or "months". Defaults to "months".

    Returns:
        float: Duration of the period in the requested unit.
    """
    since = time_range.since
    until = time_range.until

    if time_unit == "days":
        return (until - since).days + (until - since).seconds / 86400.0

    elif time_unit == "months":
        from dateutil.relativedelta import relativedelta
        rd = relativedelta(until, since)
        months = rd.years * 12 + rd.months + rd.days / 30.44
        return months

    else:
        raise ValueError("`get_duration`:time_unit must be either 'days' or 'months'")
        
@task
def download_file_and_persist(
    url: Annotated[str, Field(description="URL to download the file from")],
    output_path: Annotated[Optional[str], Field(description="Path to save the downloaded file or directory. Defaults to current working directory")] = None,
    retries: Annotated[int, Field(description="Number of retries on failure", ge=0)] = 3,
    overwrite_existing: Annotated[bool, Field(description="Whether to overwrite existing files")] = False,
    unzip: Annotated[bool, Field(description="Whether to unzip the file if it's a zip archive")] = False,
) -> str:
    """
    Downloads a file from the provided URL and persists it locally.
    If output_path is not specified, saves to the current working directory.
    Returns the full path to the downloaded file, or if unzipped, the path to the extracted directory.
    """

    if output_path is None or str(output_path).strip() == "":
        output_path = os.getcwd()
    else:
        output_path = str(output_path).strip()

    # support file:// URLs
    if output_path.startswith("file://"):
        parsed = urlparse(output_path)
        output_path = url2pathname(parsed.path)

    looks_like_dir = (
        output_path.endswith(os.sep)
        or output_path.endswith("/")
        or output_path.endswith("\\")
        or os.path.isdir(output_path)
    )

    if looks_like_dir:
        # ensure directory exists
        os.makedirs(output_path, exist_ok=True)

        # determine filename from Content-Disposition or URL
        import requests, email
        try:
            s = requests.Session()
            r = s.head(url, allow_redirects=True, timeout=10)
            cd = r.headers.get("content-disposition", "")
            filename = None
            if cd:
                # parse content-disposition safely
                m = email.message.Message()
                m["content-disposition"] = cd
                filename = m.get_param("filename")
            if not filename:
                filename = os.path.basename(urlparse(url).path.split("?")[0]) or "downloaded_file"
        except Exception:
            filename = os.path.basename(urlparse(url).path.split("?")[0]) or "downloaded_file"

        target_path = os.path.join(output_path, filename)
    else:
        target_path = output_path

    if not target_path or str(target_path).strip() == "":
        raise ValueError("Computed download target path is empty. Check 'output_path' argument.")

    # Store the parent directory to check for extracted content
    parent_dir = os.path.dirname(target_path)
    before_extraction = set()
    if unzip:
        if os.path.exists(parent_dir):
            before_extraction = set(os.listdir(parent_dir))

    # Do the download and bubble up useful context on failure
    try:
        download_file(
            url=url,
            path=target_path,
            retries=retries,
            overwrite_existing=overwrite_existing,
            unzip=unzip,
        )
    except Exception as e:
        # include debug info so callers can see what was attempted
        raise RuntimeError(
            f"download_file failed for url={url!r} path={target_path!r} retries={retries}. "
            f"Original error: {e}"
        ) from e

    # Determine the final persisted path
    if unzip and zipfile.is_zipfile(target_path):
        after_extraction = set(os.listdir(parent_dir))
        new_items = after_extraction - before_extraction
        zip_filename = os.path.basename(target_path)
        new_items.discard(zip_filename)
        
        if len(new_items) == 1:
            new_item = new_items.pop()
            new_item_path = os.path.join(parent_dir, new_item)
            if os.path.isdir(new_item_path):
                persisted_path = str(Path(new_item_path).resolve())
            else:
                persisted_path = str(Path(parent_dir).resolve())
        elif len(new_items) > 1:
            persisted_path = str(Path(parent_dir).resolve())
        else:
            extracted_dir = target_path.rsplit('.zip', 1)[0]
            if os.path.isdir(extracted_dir):
                persisted_path = str(Path(extracted_dir).resolve())
            else:
                persisted_path = str(Path(parent_dir).resolve())
    else:
        persisted_path = str(Path(target_path).resolve())

    if not os.path.exists(persisted_path):
        parent = os.path.dirname(persisted_path)
        if os.path.exists(parent):
            actual_files = os.listdir(parent)
            raise FileNotFoundError(
                f"Download failed — {persisted_path} not found after execution. "
                f"Files in {parent}: {actual_files}"
            )
        else:
            raise FileNotFoundError(
                f"Download failed — {persisted_path}. Parent dir missing: {parent}"
            )
    return persisted_path

@task
def build_mapbook_report_template(
    count: int,
    org_logo_path: Union[str, Path],
    report_period:TimeRange,
    prepared_by: str,
) -> Dict[str, str]:
    """
    Build a dictionary with the mapbook report template values.

    Args:
        count (int): Total number of subjects or records.
        org_logo_path (Union[str, Path]): Path to the organization logo file.
        report_period (TimeRange): Object with 'since', 'until', and 'time_format' attributes.
        prepared_by (str): Name of the person or organization preparing the report.

    Returns:
        Dict[str, str]: Structured dictionary with formatted metadata.
    """
    if org_logo_path.startswith("file://"):
        parsed = urlparse(org_logo_path)
        org_logo_path = url2pathname(parsed.path)

    if isinstance(org_logo_path, (str, Path)):
        org_logo_path = Path(org_logo_path)
        org_logo_path = str(org_logo_path.resolve()) if org_logo_path.exists() else str(org_logo_path)

    formatted_date = datetime.now()
    formatted_date_str = formatted_date.strftime("%Y-%m-%d %H:%M:%S")
    fmt = getattr(report_period, "time_format", "%Y-%m-%d")
    formatted_time_range = (
        f"{report_period.since.strftime(fmt)} to {report_period.until.strftime(fmt)}"
    )

    logger.info(f"Report period: {formatted_time_range}")
    logger.info(f"Report date generated: {formatted_date_str}")
    logger.info(f"Report prepared by: {prepared_by}")
    logger.info(f"Report count: {count}")
    logger.info(f"Organization logo path: {org_logo_path}")
    logger.info(f"Report ID: REP-{uuid.uuid4().hex[:8].upper()}")

    # Return structured dictionary
    return {
        "report_id": f"REP-{uuid.uuid4().hex[:8].upper()}",
        "subject_count": str(count),
        "org_logo_path": org_logo_path,
        "time_generated": formatted_date_str,
        "report_period": formatted_time_range,
        "prepared_by": prepared_by,
    }

@task
def create_context_page(
    template_path: str,
    output_directory: str,
    context: dict,
    logo_width_cm: float = 7.7,
    logo_height_cm: float = 1.93,
    filename: str = None
) -> str:
    """
    Create a context page document from a template and context dictionary.

    Args:
        template_path (str): Path to the .docx template file.
        output_directory (str): Directory to save the generated .docx file.
        context (dict): Dictionary with context values for the template.
        filename (str, optional): Optional filename for the generated file.
            If not provided, a random UUID-based filename will be generated.

    Returns:
        str: Full path to the generated .docx file.
    """
    if template_path.startswith("file://"):
        parsed = urlparse(template_path)
        template_path = url2pathname(parsed.path)

    if output_directory.startswith("file://"):
        parsed = urlparse(output_directory)
        output_directory = url2pathname(parsed.path)

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    os.makedirs(output_directory, exist_ok=True)
    if not filename:
        filename = f"context_page_{uuid.uuid4().hex}.docx"
    output_path = Path(output_directory) / filename

    doc = DocxTemplate(template_path)
    if "org_logo_path" in context and os.path.exists(context["org_logo_path"]):
        context["org_logo"] = InlineImage(
            doc,
            context["org_logo_path"],
            width=Cm(logo_width_cm),
            height=Cm(logo_height_cm),
        )

    doc.render(context)
    doc.save(output_path)
    return str(output_path)

@task
def create_mapbook_context(
    template_path: str,
    output_directory: str,
    filename: Optional[str] = None,
    subject_name: Optional[str] = None,
    time_period: Optional[TimeRange] = None,
    period: Optional[Union[int, float]] = None,
    grid_area: Optional[Union[int, float]] = None,
    mcp_area: Optional[Union[int, float]] = None,
    movement_tracks_ecomap: Optional[str] = None,
    home_range_ecomap: Optional[str] = None,
    speedmap: Optional[str] = None,
    speed_raster_ecomap: Optional[str] = None,
    night_day_ecomap: Optional[str] = None,
    seasonal_homerange: Optional[str] = None,
    validate_images: bool = True,
    box_h_cm: float = 6.5,
    box_w_cm: float = 11.11,
) -> str:
    if template_path.startswith("file://"):
        parsed = urlparse(template_path)
        template_path = url2pathname(parsed.path)

    if output_directory.startswith("file://"):
        parsed = urlparse(output_directory)
        output_directory = url2pathname(parsed.path)

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    os.makedirs(output_directory, exist_ok=True)
    if not filename:
        filename = f"mapbook_context_{uuid.uuid4().hex}.docx"
    output_path = Path(output_directory) / filename
    time_period_str = None
    if time_period:
        fmt = getattr(time_period, "time_format", "%Y-%m-%d")
        time_period_str = f"{time_period.since.strftime(fmt)} to {time_period.until.strftime(fmt)}"

    result = {}
    tpl = DocxTemplate(template_path)
    ctx = MapbookContext(
        subject_name=subject_name,
        time_period=time_period_str,
        period=period,
        grid_area=grid_area,
        mcp_area=mcp_area,
        movement_tracks_ecomap=movement_tracks_ecomap,
        home_range_ecomap=home_range_ecomap,
        speedmap=speedmap,
        speed_raster_ecomap=speed_raster_ecomap,
        night_day_ecomap=night_day_ecomap,
        seasonal_homerange=seasonal_homerange,
    )
    if validate_images:
        for field_name, value in asdict(ctx).items():
            if isinstance(value, str) and Path(value).suffix.lower() in (".png", ".jpg", ".jpeg"):
                p = Path(value)
                if not p.exists() or not p.is_file():
                    warnings.warn(f"Image for '{field_name}' not found or not a file: {value}")

    base = asdict(ctx)

    for key, value in base.items():
        if isinstance(value, str) and Path(value).suffix.lower() in (".png", ".jpg", ".jpeg"):
            result[key] = InlineImage(tpl, value, width=Cm(box_w_cm), height=Cm(box_h_cm))
        else:
            result[key] = value

    tpl.render(result)
    tpl.save(output_path)
    return str(output_path)

def _fallback_to_none_doc(
    obj: tuple[CompositeFilter | None, str] | SkipSentinel
    ) -> tuple[CompositeFilter | None, str] | None:
    return None if isinstance(obj, SkipSentinel) else obj

@dataclass
class GroupedDoc:
    """Analogous to GroupedWidget but for document pages."""
    views: dict[CompositeFilter | None, Optional[str]]

    @classmethod
    def from_single_view(cls, item: tuple[CompositeFilter | None, str]) -> "GroupedDoc":
        view, path = item
        return cls(views={view: path})

    @property
    def merge_key(self) -> str:
        """
        Determine how docs should be grouped.
        Default: group by filename stem of the first non-None path in views.
        If you want another grouping (e.g. based on metadata), replace this logic.
        """
        # pick any path available
        for p in self.views.values():
            if p:
                return Path(p).stem
        # fallback unique key (shouldn't happen normally)
        return uuid.uuid4().hex

    def __ior__(self, other: "GroupedDoc") -> "GroupedDoc":
        """Merge views from other into self. Keys must be compatible by merge_key."""
        if self.merge_key != other.merge_key:
            raise ValueError(f"Cannot merge GroupedDoc with different keys: {self.merge_key} != {other.merge_key}")
        # update views (later views override same view key)
        self.views.update(other.views)
        return self

@task
def combine_docx_files(
    cover_page_path: Annotated[str, Field(description="Path to the cover page .docx file")],
    context_page_items: Annotated[
        list[
            Annotated[
                tuple[CompositeFilter | None, str],
                SkippedDependencyFallback(_fallback_to_none_doc),
            ]
        ],
        Field(description="List of context pages. Items can be SkipSentinel and will be filtered out.", exclude=True),
    ],
    output_directory: Annotated[str, Field(description="Directory where combined docx will be written")],
    filename: Annotated[Optional[str], Field(description="Optional output filename")] = None,
) -> Annotated[str, Field(description="Path to the combined .docx file")]:
    """
    Combine cover + grouped context pages into a single DOCX.
    """
    from docx import Document
    from docxcompose.composer import Composer
    
    valid_items = [it for it in context_page_items if it is not None]
    grouped_docs = [GroupedDoc.from_single_view(it) for it in valid_items]

    merged_map: dict[str, GroupedDoc] = {}
    for gd in grouped_docs:
        key = gd.merge_key
        if key not in merged_map:
            merged_map[key] = gd
        else:
            merged_map[key] = gd

    final_paths: list[str] = []
    for group in merged_map.values():
        for view_key, p in group.views.items():
            if p is not None:
                final_paths.append(p)

    if not os.path.exists(cover_page_path):
        raise FileNotFoundError(f"Cover page file not found: {cover_page_path}")
        
    for p in final_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Context page file not found: {p}")
    
    if output_directory.startswith("file://"):
        parsed = urlparse(output_directory)
        output_directory = url2pathname(parsed.path)

    os.makedirs(output_directory, exist_ok=True)

    if not filename:
        filename = f"overall_mapbook_{uuid.uuid4().hex}.docx"
    output_path = Path(output_directory) / filename

    master = Document(cover_page_path)
    composer = Composer(master)
    for doc_path in final_paths:
        doc = Document(doc_path)
        composer.append(doc) 
        
    composer.save(output_path)
    return str(output_path)

@task
def round_off_values(value: float, dp: int) -> float:
    return round(value, dp)