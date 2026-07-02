import numpy as np
from typing import Optional
from wt_registry import register
from ecoscope.base.utils import hex_to_rgba
from ecoscope.platform.annotations import AnyDataFrame


@register()
def add_status_color_columns(
    gdf: AnyDataFrame,
    hex_column: str,
    previous_color_hex: str,
    use_hex_column_for_current: bool = True,
    default_current_hex: Optional[str] = None,
    current_status: str = "Current tracks",
    status_column: str = "duration_status",
) -> AnyDataFrame:
    """
    Assign hex and RGBA colors for a status column (current vs. everything else).

    Adds two columns (named from `status_column`):
        - {status_column}_hex_colors: hex color codes
        - {status_column}_colors:     RGBA tuples

    Args:
        current_status: The value in `status_column` treated as "current".
        status_column: Column holding the status labels.
    """
    out = gdf.copy()

    # Choose the color source for "current" rows.
    if use_hex_column_for_current:
        current_hex = out[hex_column]
    elif default_current_hex is not None:
        current_hex = default_current_hex
    else:
        # Consider raising here instead of silently reusing previous_color_hex.
        current_hex = previous_color_hex

    is_current = out[status_column] == current_status

    hex_col_name = f"{status_column}_hex_colors"
    rgba_col_name = f"{status_column}_colors"

    out[hex_col_name] = np.where(is_current, current_hex, previous_color_hex)

    # hex_to_rgba only needs to run once per distinct color.
    rgba_lookup = {h: hex_to_rgba(h) for h in out[hex_col_name].unique()}
    out[rgba_col_name] = out[hex_col_name].map(rgba_lookup)
    return out
