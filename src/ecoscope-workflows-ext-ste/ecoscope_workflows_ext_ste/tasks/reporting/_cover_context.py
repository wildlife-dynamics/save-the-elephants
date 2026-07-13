import os
from PIL import Image
from pathlib import Path
from pydantic import Field
from docx.shared import Inches
from datetime import datetime
from wt_registry import register
from docxtpl import DocxTemplate, InlineImage
from ecoscope.platform.tasks.filter._filter import TimeRange
from typing import Any, Dict, Mapping, Optional, Union, Annotated
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme


def get_image_dimensions_from_pixels(
    image_path: str,
    dpi: int = 96,
    max_dimension_inches: float = 1.5,
) -> tuple[float, float]:
    """
    Calculate image dimensions in inches based on pixel dimensions,
    scaled so the largest dimension fits within max_dimension_inches.
    Preserves aspect ratio for both wide and square images.
    """
    with Image.open(image_path) as img:
        width_pixels, height_pixels = img.size

        # Get DPI from image metadata
        image_dpi = img.info.get("dpi", (dpi, dpi))
        if isinstance(image_dpi, tuple):
            dpi_x, dpi_y = image_dpi
        else:
            dpi_x = dpi_y = dpi

    width_inches = width_pixels / dpi_x
    height_inches = height_pixels / dpi_y

    if width_inches > height_inches:
        scale_factor = max_dimension_inches / width_inches
    else:
        scale_factor = max_dimension_inches / height_inches

    return width_inches * scale_factor, height_inches * scale_factor


@register()
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
    filename: Annotated[
        Optional[str],
        Field(
            description="Optional filename.",
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
        filename (str, optional): Optional filename for the generated file.
            If not provided, a random UUID-based filename will be generated.

    Returns:
        str: Full path to the generated .docx file.
    """
    # Normalize paths
    template_path = remove_file_scheme(template_path)
    output_dir = remove_file_scheme(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        filename = "cover_page.docx"
    output_path = Path(output_dir) / filename

    doc = DocxTemplate(template_path)
    if "org_logo_path" in context and os.path.exists(context["org_logo_path"]):
        width, height = get_image_dimensions_from_pixels(
            context["org_logo_path"],
            dpi=125,
            max_dimension_inches=1.5,  # Adjust this value as needed
        )
        context["org_logo"] = InlineImage(
            doc,
            context["org_logo_path"],
            width=Inches(width),
            height=Inches(height),
        )
    doc.render(context)
    doc.save(output_path)
    print(f"create_context_page: saved to {output_path}")
    return str(output_path)


@register()
def prepare_cover_metadata(
    org_logo_path: Union[str, Path, None],
    report_period: TimeRange,
    prepared_by: str,
    extra_fields: Optional[Mapping[str, Any]] = None,
    time_generated_format: str = "%Y-%m-%d %H:%M:%S",
) -> Dict[str, Optional[str]]:
    """
    Build the context dictionary of template values for a report cover.

    Always produces the four core fields every cover needs:
        - time_generated
        - org_logo_path
        - report_period
        - prepared_by

    Any report-specific values (subject counts, report IDs, titles, etc.)
    are passed via `extra_fields` and merged into the result, so this stays
    reusable across report types without signature changes.

    Args:
        org_logo_path: Path to the org logo, or None if unavailable.
        report_period: Object with 'since', 'until', and optional
            'time_format' attributes.
        prepared_by: Person or organization preparing the report.
        extra_fields: Additional template values to merge in. Values are
            stringified for template consumption. Keys here override the
            core fields if they collide.
        time_generated_format: strftime format for the generation timestamp.

    Returns:
        Dict of formatted metadata for the cover template.
    """
    resolved_logo_path: Optional[str] = None
    if org_logo_path is not None:
        resolved_logo_path = remove_file_scheme(org_logo_path)
        if not resolved_logo_path.strip():
            raise ValueError("org_logo_path is empty after normalization")

    fmt = getattr(report_period, "time_format", "%Y-%m-%d")
    formatted_time_range = f"{report_period.since.strftime(fmt)} to " f"{report_period.until.strftime(fmt)}"

    ctx: Dict[str, Optional[str]] = {
        "time_generated": datetime.now().strftime(time_generated_format),
        "org_logo_path": resolved_logo_path,  # None if no logo was provided
        "report_period": formatted_time_range,
        "prepared_by": prepared_by,
    }

    # Merge report-specific values, stringifying so templates get strings.
    if extra_fields:
        for key, value in extra_fields.items():
            ctx[key] = None if value is None else str(value)

    print(f"prepare_cover_metadata: result={ctx}")
    return ctx


@register()
def build_extra_fields(fields: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Package arbitrary report-specific values into an `extra_fields` dict.

    `fields` is a mapping of name -> value; each caller supplies whatever
    fields apply to its report. Values are stringified (None preserved).
    """
    print(f"build_extra_fields: input fields={dict(fields)}")
    result = {key: (None if value is None else str(value)) for key, value in fields.items()}
    print(f"build_extra_fields: result={result}")
    return result
