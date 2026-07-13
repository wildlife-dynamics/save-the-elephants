import numpy as np
from typing import Optional, cast
from wt_registry import register
from ecoscope.base.utils import hex_to_rgba
from ecoscope.platform.annotations import AnyDataFrame


@register()
def add_status_color_columns(
    df: AnyDataFrame,
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
    out = df.copy()

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


@register()
def add_rgba_from_hex(
    df: AnyDataFrame,
    column: str,
    new_column: str,
) -> AnyDataFrame:
    """
    Add a column of RGBA tuples derived from a column of hex color strings.

    Args:
        df: Input DataFrame containing the source hex color column.
        column: Name of the column containing hex color strings (e.g. "#fcb5ac").
        new_column: Name of the new column to store RGBA tuples.

    Returns:
        A new DataFrame with the added RGBA column.

    Raises:
        ValueError: If `column` is not a column in `df`.
    """
    print(f"[add_rgba_from_hex] Converting hex colors in '{column}' -> RGBA tuples in '{new_column}'")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")

    df = df.copy()
    df[new_column] = df[column].apply(hex_to_rgba)
    print(f"[add_rgba_from_hex] Converted {df[column].nunique()} unique hex color(s) across {len(df)} rows")
    return cast(AnyDataFrame, df)
