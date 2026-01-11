import os
import uuid
from pathlib import Path
from pydantic import Field
from docx.shared import Cm
from datetime import datetime
from docxtpl import DocxTemplate, InlineImage
from ecoscope_workflows_core.decorators import task
from typing import Annotated, Optional, Dict, Any, Union
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_core.tasks.transformation._unit import Quantity
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme


@task
def create_context_page(
    template_path: Annotated[
        str,
        Field(
            description="Path to the .docx template file.",
        ),
    ],
    output_dir: Annotated[
        str,
        Field(
            description="Directory to save the generated .docx file.",
        ),
    ],
    context: Annotated[
        dict,
        Field(
            description="Dictionary with context values for the template.",
        ),
    ],
    logo_width_cm: Annotated[
        float,
        Field(
            description="Width of the logo in centimeters.",
        ),
    ] = 7.7,
    logo_height_cm: Annotated[
        float,
        Field(
            description="Height of the logo in centimeters.",
        ),
    ] = 1.93,
    filename: Annotated[
        Optional[str],
        Field(
            description="Optional filename . If not provided, a random UUID-based filename will be generated.",
            exclude=True,
        ),
    ] = None,
) -> Annotated[
    str,
    Field(
        description="Full path to the generated .docx file.",
    ),
]:
    """
    Create a context page document from a template and context dictionary.

    Args:
        template_path (str): Path to the .docx template file.
        output_dir (str): Directory to save the generated .docx file.
        context (dict): Dictionary with context values for the template.
        logo_width_cm (float): Width of the logo in centimeters. Default is 7.7.
        logo_height_cm (float): Height of the logo in centimeters. Default is 1.93.
        filename (str, optional): Optional filename for the generated file.
            If not provided, a random UUID-based filename will be generated.

    Returns:
        str: Full path to the generated .docx file.
    """
    # Normalize paths
    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)

    # Validate paths
    if not template_path.strip():
        raise ValueError("template_path is empty after normalization")
    if not output_dir.strip():
        raise ValueError("output_dir is empty after normalization")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        filename = "context_page_.docx"
    output_path = Path(output_dir) / filename

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
def create_mapbook_ctx_cover(
    count: int,
    org_logo_path: Union[str, Path],
    report_period: TimeRange,
    prepared_by: str,
) -> Dict[str, str]:
    """
    Build a dictionary with the mapbook report template values.

    Args:
        count (int): Total number of subjects or records.
        report_period (TimeRange): Object with 'since', 'until', and 'time_format' attributes.
        prepared_by (str): Name of the person or organization preparing the report.

    Returns:
        Dict[str, str]: Structured dictionary with formatted metadata.
    """
    org_logo_path = remove_file_scheme(org_logo_path)

    if not org_logo_path.strip():
        raise ValueError("org_logo_path is empty after normalization")
    formatted_date = datetime.now()
    formatted_date_str = formatted_date.strftime("%Y-%m-%d %H:%M:%S")
    fmt = getattr(report_period, "time_format", "%Y-%m-%d")
    formatted_time_range = f"{report_period.since.strftime(fmt)} to {report_period.until.strftime(fmt)}"

    # Return structured dictionary
    return {
        "report_id": f"REP-{uuid.uuid4().hex[:8].upper()}",
        "subject_count": str(count),
        "org_logo_path": org_logo_path,
        "time_generated": formatted_date_str,
        "report_period": formatted_time_range,
        "prepared_by": prepared_by,
    }


def validate_image_path(field_name: str, path: str) -> None:
    """Validate that an image file exists and has valid extension."""
    normalized_path = remove_file_scheme(path)

    if not os.path.exists(normalized_path):
        raise FileNotFoundError(f"Image file for '{field_name}' not found: {normalized_path}")

    valid_extensions = {".png", ".jpg", ".jpeg"}
    if Path(normalized_path).suffix.lower() not in valid_extensions:
        raise ValueError(
            f"Invalid image format for '{field_name}': {Path(normalized_path).suffix}. "
            f"Expected one of {valid_extensions}"
        )

    print(f" Validated image for '{field_name}': {normalized_path}")


@task
def create_mapbook_grouper_ctx(
    current_period: TimeRange,
    previous_period: TimeRange,
    period: float | None,
    grouper_name: list,
    grid_area: Quantity | None,
    mcp_area: Quantity | None,
    map_paths: list | None,
) -> Dict[str, str]:
    TIME_FMT = "%d %b %Y %H:%M:%S"

    # Format current period
    current_period_str = None
    if current_period:
        current_period_str = (
            f"{current_period.since.strftime(TIME_FMT)} " f"to {current_period.until.strftime(TIME_FMT)}"
        )

    # Format previous period
    previous_period_str = None
    if previous_period:
        previous_period_str = (
            f"{previous_period.since.strftime(TIME_FMT)} " f"to {previous_period.until.strftime(TIME_FMT)}"
        )

    # Format period
    period_str = f"{period:.2f}" if isinstance(period, float) else str(period)

    # Format areas with None handling
    grid_area_str = "N/A"
    if grid_area and grid_area.value is not None:
        grid_area_str = f"{grid_area.value:.1f} {grid_area.unit}"

    mcp_area_str = "N/A"
    if mcp_area and mcp_area.value is not None:
        mcp_area_str = f"{mcp_area.value:.1f} {mcp_area.unit}"

    print(f"grid area: {grid_area_str} || mcp area: {mcp_area_str}")

    # Handle grouper value
    grouper_value = "All"
    print("grouper name: {grouper_name}")
    if isinstance(grouper_name, list) and len(grouper_name) > 0:
        grouper_value = str(grouper_name[0])
    print(f"grouper_value: {grouper_value}")

    # Map parsing with None handling
    map_suffixes = {
        "movement_tracks_map": "movement_tracks.png",
        "home_range_map": "homerange.png",
        "speed_map": "speedmap.png",
        "speed_raster_map": "mean_speed_raster.png",
        "night_day_ecomap": "day_night.png",
        "seasonal_map": "seasonal_home_range.png",
    }

    mapbook_png_paths = {key: None for key in map_suffixes.keys()}

    if map_paths:
        for path in map_paths:
            if path:  # Check path is not None
                for key, suffix in map_suffixes.items():
                    if path.endswith(suffix):
                        mapbook_png_paths[key] = path
                        break

    # Build context
    ctx = {
        "time_period": current_period_str,
        "previous_time_range": previous_period_str,
        "period": period_str,
        "grid_area": grid_area_str,
        "mcp_area": mcp_area_str,
        "grouper_value": grouper_value,
        **mapbook_png_paths,
    }

    print(f"Context: {ctx}")
    return ctx


def prepare_context_for_template(
    context: Any,
    template: DocxTemplate,
    box_h_cm: float = 6.5,
    box_w_cm: float = 11.11,
) -> dict:
    result = {}

    for key, value in context.items():
        if value is None:
            result[key] = value
            continue

        # Handle image fields - check if it's a valid image path
        if isinstance(value, str):
            normalized_path = remove_file_scheme(value)
            if os.path.exists(normalized_path):
                path_obj = Path(normalized_path)
                # If it's an image file, convert to InlineImage
                if path_obj.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    result[key] = InlineImage(template, normalized_path, width=Cm(box_w_cm), height=Cm(box_h_cm))
                else:
                    result[key] = value
            else:
                result[key] = value
        # Handle numeric fields - convert to integers
        elif isinstance(value, (int, float)):
            result[key] = int(value)
        else:
            result[key] = value

    return result


@task
def create_grouper_page(
    template_path: str,
    output_dir: str,
    context: dict[str, Any],
    filename: Optional[str] = None,
    validate_images: bool = True,
    box_h_cm: float = 6.5,
    box_w_cm: float = 11.11,
) -> str:
    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)

    # Validate paths
    if not template_path.strip():
        raise ValueError("template_path is empty after normalization")
    if not output_dir.strip():
        raise ValueError("output_directory is empty after normalization")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    if not filename:
        filename = f"context_{uuid.uuid4().hex}.docx"
    output_path = Path(output_dir) / filename

    if validate_images:
        for field_name, value in context.items():
            if isinstance(value, str) and Path(value).suffix.lower() in (".png", ".jpg", ".jpeg"):
                validate_image_path(field_name, value)

    try:
        tpl = DocxTemplate(template_path)
    except Exception as e:
        raise ValueError(f"Failed to load template: {e}")

    rendered_context = prepare_context_for_template(
        context=context,
        template=tpl,
        box_h_cm=box_h_cm,
        box_w_cm=box_w_cm,
    )
    try:
        tpl.render(rendered_context)
        tpl.save(output_path)
        return str(output_path)
    except Exception as e:
        raise ValueError(f"Failed to render or save document: {e}")


# merge word docs
@task
def merge_mapbook_files(
    cover_page_path: Annotated[str, Field(description="Path to the cover page .docx file")],
    context_page_items: Annotated[list[Any], Field(description="List of context page document paths to merge.")],
    output_dir: Annotated[str, Field(description="Directory where combined docx will be written")],
    filename: Annotated[Optional[str], Field(description="Optional output filename")] = None,
) -> Annotated[str, Field(description="Path to the combined .docx file")]:
    """
    Combine cover + context pages into a single DOCX.
    Orders context pages by calendar month if month names are detected.
    """
    import os
    import calendar
    from pathlib import Path
    from docx import Document
    from docxcompose.composer import Composer

    # Build month lookup from calendar module
    MONTH_LOOKUP = {name.lower(): idx for idx, name in enumerate(calendar.month_name) if name}
    MONTH_ABBR_LOOKUP = {name.lower(): idx for idx, name in enumerate(calendar.month_abbr) if name}

    def extract_path(item, idx):
        if isinstance(item, str):
            return item
        if isinstance(item, (list, tuple)):
            for x in item:
                if isinstance(x, str) and os.path.exists(x):
                    return x
            for x in reversed(item):
                if isinstance(x, str):
                    return x
            raise ValueError(f"context_page_items[{idx}] is a list/tuple but contains no string path: {item!r}")
        raise ValueError(f"context_page_items[{idx}] has unsupported type {type(item)}; expected str or tuple/list")

    def detect_month(path: str):
        """Detect month name or abbreviation in filename/path."""
        name = os.path.basename(path).lower()

        for month, idx in MONTH_LOOKUP.items():
            if month in name:
                return idx

        for abbr, idx in MONTH_ABBR_LOOKUP.items():
            if abbr in name:
                return idx

        return None

    # Normalize paths
    normalized_paths = []
    print(f"Context page items: {context_page_items}")
    for idx, item in enumerate(context_page_items):
        if item is None:
            continue
        normalized_paths.append(extract_path(item, idx))

    if not os.path.exists(cover_page_path):
        raise FileNotFoundError(f"Cover page file not found: {cover_page_path}")

    for p in normalized_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Context page file not found: {p}")

    # Calendar-aware ordering
    with_month = []
    without_month = []

    for i, path in enumerate(normalized_paths):
        month_idx = detect_month(path)
        if month_idx is not None:
            with_month.append((month_idx, path))
        else:
            without_month.append((i, path))  # preserve original order

    with_month.sort(key=lambda x: x[0])
    ordered_paths = [p for _, p in with_month] + [p for _, p in without_month]

    output_dir = remove_file_scheme(output_dir)
    if not output_dir.strip():
        raise ValueError("output_dir is empty after normalization")

    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"overall_report_{timestamp}.docx"

    output_path = Path(output_dir) / filename

    master = Document(cover_page_path)
    composer = Composer(master)

    for doc_path in ordered_paths:
        doc = Document(doc_path)
        composer.append(doc)

    composer.save(str(output_path))
    return str(output_path)
