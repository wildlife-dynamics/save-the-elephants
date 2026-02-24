# read outputs on output dir, pass output dir, df to get name and pass list of seasonal homerange images
import os
import re
from pathlib import Path
from typing import Optional
from docx.shared import Inches
from docxtpl import DocxTemplate, InlineImage
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme


def build_report_context(output_dir: str) -> dict:
    filename_map = {
        "collared_points": "collared_elephants_map",
        "movement_tracks": "historical_current_tracks_map",
        "subject_tracks": "combined_subject_tracks_map",
        "overall_homerange": "home_range_metrics_map",
        "filtered_homerange": "full_home_range_metrics_map",
        "time_of_day_dominance": "night_day_raster_map",
        "night_fixes": "night_time_fixes_map",
        "dry_mean_speed_raster_map": "dry_speed_raster_map",
        "wet_mean_speed_raster_map": "wet_speed_raster_map",
        "recursion_events": "recursion_events_map",
        "protection_status_bar": "proportion_of_fixes_bar_chart",
        "protected_areas": "protected_areas_home_range_map",
        "unprotected_areas": "unprotected_areas_home_range_map",
    }

    ctx = {key: None for key in filename_map.values()}
    ctx["seasonal_images"] = []

    seasonal_pattern = re.compile(r"^[0-9a-f]+_(.+)\.png$")

    for png_path in Path(output_dir).rglob("*.png"):
        stem = png_path.stem
        full_path = str(png_path)

        if stem in filename_map:
            ctx[filename_map[stem]] = full_path
        else:
            match = seasonal_pattern.match(png_path.name)
            if match:
                subject_name = match.group(1)
                ctx["seasonal_images"].append({"name": subject_name, "image": full_path})

    return ctx


def prepare_general_context(
    context,
    template,
    height: float = 6.5,
    width: float = 11.11,
):
    def transform(value):
        if value is None:
            return value

        if isinstance(value, dict):
            return {k: transform(v) for k, v in value.items()}

        if isinstance(value, list):
            return [transform(v) for v in value]

        if isinstance(value, str):
            normalized_path = remove_file_scheme(value)

            if os.path.exists(normalized_path):
                path_obj = Path(normalized_path)

                if path_obj.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    return InlineImage(
                        template,
                        normalized_path,
                        width=Inches(width),
                        height=Inches(height),
                    )

            return value
        if isinstance(value, (int, float)):
            return int(value)
        return value

    return transform(context)


def build_panel_rows(images, cols=2):
    rows = []
    for i in range(0, len(images), cols):
        row = images[i : i + cols]

        # pad empty cells so table doesn't break
        while len(row) < cols:
            row.append({"name": "", "image": ""})

        rows.append(row)

    return rows


@task
def general_template_context(
    output_dir: str, template_path: str, filename: Optional[str] = None, width: float = 5.34, height: float = 3.12
) -> str:
    output_dir = remove_file_scheme(output_dir)
    template_path = remove_file_scheme(template_path)

    # Validate paths
    if not template_path.strip():
        raise ValueError("template_path is empty after normalization")
    if not output_dir.strip():
        raise ValueError("output_dir is empty after normalization")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    context = build_report_context(output_dir)
    if not filename:
        filename = "general_context.docx"
    output_path = Path(output_dir) / filename

    context["seasonal_panel"] = build_panel_rows(context["seasonal_images"], cols=2)
    del context["seasonal_images"]

    try:
        tpl = DocxTemplate(template_path)
    except Exception as e:
        raise ValueError(f"Failed to load template: {e}")

    ctx = prepare_general_context(context=context, template=tpl, height=height, width=width)
    try:
        tpl.render(ctx)
        tpl.save(output_path)
        return str(output_path)
    except Exception as e:
        raise ValueError(f"Failed to render or save document: {e}")
