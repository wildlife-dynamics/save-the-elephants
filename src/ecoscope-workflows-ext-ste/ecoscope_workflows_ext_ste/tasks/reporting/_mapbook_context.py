import os
import re
import uuid
from PIL import Image
from pathlib import Path
from docx.shared import Cm
from wt_registry import register
from docxtpl import DocxTemplate, InlineImage
from typing import Dict, Optional, Mapping, Any
from wt_task.skip import SkipSentinel, SKIP_SENTINEL
from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.tasks.filter._filter import TimeRange
from ecoscope.platform.tasks.transformation._unit import Quantity
from ..transformation import safe_string
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

TIME_FMT = "%d %b %Y %H:%M:%S"
DEFAULT_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})
MAP_SUFFIXES = {
    "movement_tracks_map": "_movement_tracks",
    "home_range_map": "_homerange",
    "speed_map": "_speedmap",
    "speed_raster_map": "_mean_speed_raster",
    "night_day_ecomap": "_day_night",
    "seasonal_map": "_seasonal_homerange",
}


def _unwrap_skip(value):
    """Flatten SkipSentinel (and skip-containing containers) to None/values."""
    if value is None or value is SKIP_SENTINEL:
        return None
    if isinstance(value, (list, tuple)):
        items = [v for v in (_unwrap_skip(v) for v in value) if v is not None]
        if not items:
            return None
        return items[0] if len(items) == 1 else type(value)(items)
    return value


def _format_area(area) -> str:
    if area is not None and getattr(area, "value", None) is not None:
        return f"{area.value:.1f} {area.unit}"
    return "N/A"


def _format_period(tr) -> str:
    if tr is None:
        return "N/A"
    return f"{tr.since.strftime(TIME_FMT)} to {tr.until.strftime(TIME_FMT)}"


def _find_map_file(root: Path, safe_name: str, suffix: str) -> Optional[str]:
    """Find `{safe_name}{suffix}.{ext}` under root, preferring png over html."""
    for ext in DEFAULT_IMAGE_EXTENSIONS:
        # exact match first (this is what persist_text/persist_df wrote)
        exact = root / f"{safe_name}{suffix}{ext}"
        if exact.exists():
            return str(exact)
        # fall back to glob in case a suffix was appended (e.g. dedup counters)
        matches = sorted(root.glob(f"{safe_name}{suffix}*{ext}"))
        if matches:
            if len(matches) > 1:
                print(
                    f"Multiple candidates for {safe_name}{suffix}{ext}: {matches}; "
                    f"using the most recently modified."
                )
                matches.sort(key=lambda p: p.stat().st_mtime)
            return str(matches[-1])
    return None


@register()
def create_mapbook_context(
    df: AnyDataFrame | SkipSentinel | None,
    current_period: TimeRange | SkipSentinel | None,
    previous_period: TimeRange | SkipSentinel | None,
    period: float | SkipSentinel | None,
    grid_area: Quantity | SkipSentinel | None,
    mcp_area: Quantity | SkipSentinel | None,
    root_path: str,  # pass ${{ env.ECOSCOPE_WORKFLOWS_RESULTS }}
) -> Dict[str, Optional[str]]:
    df = _unwrap_skip(df)
    current_period = _unwrap_skip(current_period)
    previous_period = _unwrap_skip(previous_period)
    period = _unwrap_skip(period)
    grid_area = _unwrap_skip(grid_area)
    mcp_area = _unwrap_skip(mcp_area)

    grouper_value = "All"
    safe_name = None

    if df is not None and "subject_name" in df.columns:
        unique_names = df["subject_name"].dropna().unique()
        if len(unique_names) == 1:
            grouper_value = str(unique_names[0])
            safe_name = safe_string(grouper_value)
        elif len(unique_names) > 1:
            # Grouped by something other than subject_name (e.g. sex/subtype):
            # per-subject file naming does not apply.
            print(f"{len(unique_names)} subject names in group; " f"cannot resolve per-subject map files.")
    else:
        print("df is None or missing 'subject_name'; using grouper_value='All'")

    print(f"grouper_value={grouper_value!r}, safe_name={safe_name!r}")

    root = Path(root_path)
    mapbook_png_paths = {}
    for ctx_key, suffix in MAP_SUFFIXES.items():
        path = _find_map_file(root, safe_name, suffix) if safe_name else None
        mapbook_png_paths[ctx_key] = path if path else "N/A"
        if path is None:
            print(f"No file found for {ctx_key} ({safe_name}{suffix})")

    ctx = {
        "current_period": _format_period(current_period),
        "previous_period": _format_period(previous_period),
        "period": f"{period:.2f}" if isinstance(period, float) else "N/A",
        "grid_area": _format_area(grid_area),
        "mcp_area": _format_area(mcp_area),
        "grouper_value": grouper_value,
        **mapbook_png_paths,
    }

    print(f"Context for {grouper_value}: {ctx}")
    return ctx


def _fit_within_box(image_path: str, box_w_cm: float, box_h_cm: float) -> tuple[float, float]:
    """Scale native image dims to fit inside the box, preserving aspect ratio."""
    with Image.open(image_path) as im:
        px_w, px_h = im.size
    scale = min(box_w_cm / px_w, box_h_cm / px_h)
    return px_w * scale, px_h * scale


def _make_inline_image(template: DocxTemplate, path: str, box_w_cm: float, box_h_cm: float) -> Optional[InlineImage]:
    try:
        w_cm, h_cm = _fit_within_box(path, box_w_cm, box_h_cm)
        return InlineImage(template, path, width=Cm(w_cm), height=Cm(h_cm))
    except Exception as exc:  # corrupt/unreadable image shouldn't kill the render
        print(f"Could not embed image {path}: {exc}")
        return None


def build_docx_context(
    context: Mapping,
    template: DocxTemplate,
    box_h_cm: float = 6.5,
    box_w_cm: float = 11.11,
) -> dict:
    """Convert a raw context mapping into a docxtpl-ready rendering context.

    - Any string value pointing at an existing image file becomes an
      InlineImage fitted inside the box.
    - None -> "", bools -> Yes/No, numbers -> display-formatted strings.
    - All other values pass through unchanged. No keys are added or removed.
    """
    result = {}

    for key, value in context.items():
        if value is None:
            result[key] = ""

        elif isinstance(value, str):
            normalized = remove_file_scheme(value)
            if os.path.exists(normalized) and Path(normalized).suffix.lower() in DEFAULT_IMAGE_EXTENSIONS:
                image = _make_inline_image(template, normalized, box_w_cm, box_h_cm)
                result[key] = image if image is not None else ""
            else:
                result[key] = value

        elif isinstance(value, bool):
            result[key] = "Yes" if value else "No"
        elif isinstance(value, int):
            result[key] = f"{value:,}"
        elif isinstance(value, float):
            result[key] = f"{value:,.2f}"
        else:
            result[key] = value

    return result


def _default_filename(context: dict[str, Any]) -> str:
    """Name the page after the grouper value; UUID only as a last resort."""
    grouper_value = str(context.get("grouper_value", "")).strip()
    if grouper_value and grouper_value.upper() != "N/A":
        safe = re.sub(r"[^\w\-]+", "_", grouper_value).strip("_")
        if safe:
            return f"{safe}.docx"
    return f"{uuid.uuid4().hex[:8]}.docx"


@register()
def render_mapbook_page(
    template_path: str,
    output_dir: str,
    context: dict[str, Any],
    filename: str,
    strict_images: bool = False,
    box_h_cm: float = 6.5,
    box_w_cm: float = 11.11,
) -> str:
    """Render one mapbook page from a docx template and return its path.

    strict_images=False (default): missing/unreadable images are logged and
    rendered as blank slots. strict_images=True: they raise instead.
    """
    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)

    if not template_path.strip():
        raise ValueError("template_path is empty after normalization")
    if not output_dir.strip():
        raise ValueError("output_dir is empty after normalization")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        filename = _default_filename(context)
    output_path = Path(output_dir) / filename

    # --- image validation: warn by default, raise only in strict mode ---
    for field_name, value in context.items():
        if not isinstance(value, str) or not value:
            continue
        normalized = remove_file_scheme(value)
        if Path(normalized).suffix.lower() in DEFAULT_IMAGE_EXTENSIONS:
            if not os.path.exists(normalized):
                msg = f"{field_name}: image not found: {normalized}"
                if strict_images:
                    raise FileNotFoundError(msg)
                print(msg)

    try:
        tpl = DocxTemplate(template_path)
    except Exception as e:
        raise ValueError(f"Failed to load template {template_path}: {e}") from e

    rendered_context = build_docx_context(
        context=context,
        template=tpl,
        box_h_cm=box_h_cm,
        box_w_cm=box_w_cm,
    )

    try:
        tpl.render(rendered_context)
        tpl.save(output_path)
    except Exception as e:
        raise ValueError(
            f"Failed to render or save {output_path} " f"(grouper={context.get('grouper_value', '?')}): {e}"
        ) from e

    print(f"Rendered mapbook page: {output_path}")
    return str(output_path)
