from typing import Optional
from ecoscope.base.utils import hex_to_rgba
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyDataFrame


@task
def modify_status_colors(grouper_value: str, gdf: AnyDataFrame) -> AnyDataFrame:
    """
    Modify colors based on grouper value and duration status.

    Args:
        grouper_value: The grouping column (e.g., 'subject_name', 'subject_sex')
        gdf: GeoDataFrame with duration_status and hex_color columns

    Returns:
        GeoDataFrame with assigned duration status colors
    """
    if not grouper_value or grouper_value.strip() == "":
        raise ValueError("`modify_status_colors`: grouper_value is empty.")

    if grouper_value == "subject_name":
        # Use individual subject colors for current, gray for previous
        gdf = assign_status_colors(
            gdf=gdf,
            hex_column="hex_color",
            previous_color_hex="#808080",  # Gray for previous
            use_hex_column_for_current=True,
        )
    else:  # Covers "subject_sex", "subject_subtype", and everything else
        # Use uniform dark blue for current, gray for previous
        gdf = assign_status_colors(
            gdf=gdf,
            hex_column="hex_color",
            previous_color_hex="#808080",  # Gray for previous
            use_hex_column_for_current=False,
            default_current_hex="#00008b",  # Dark blue for current
        )
    return gdf


@task
def assign_status_colors(
    gdf: AnyDataFrame,
    hex_column: str,
    previous_color_hex: str,
    use_hex_column_for_current: bool = True,
    default_current_hex: Optional[str] = None,
) -> AnyDataFrame:
    """
    Assign hex and RGBA colors for duration status (Current/Previous).

    Args:
        gdf: GeoDataFrame/DataFrame with duration_status and hex column.
        hex_column: Column containing hex color codes for current period.
        previous_color_hex: Hex color for "Previous" duration status.
        use_hex_column_for_current:
            If True  -> use gdf[hex_column] for "Current" status.
            If False -> use default_current_hex for "Current" status.
        default_current_hex:
            Optional hex color for "Current" when use_hex_column_for_current=False.

    Returns:
        GeoDataFrame with two new columns:
        - duration_status_hex_colors: hex color codes
        - duration_status_colors: RGBA tuples
    """
    if gdf is None or gdf.empty:
        raise ValueError("`assign_status_colors`: gdf is empty.")

    df = gdf.copy()
    if not isinstance(previous_color_hex, str) or not previous_color_hex.startswith("#"):
        raise ValueError("`assign_status_colors`: Invalid previous_color_hex.")
    if hex_column not in df.columns:
        raise ValueError(f"`assign_status_colors`: Missing column {hex_column!r}")
    if "duration_status" not in df.columns:
        raise ValueError("`assign_status_colors`: Missing 'duration_status' column.")

    if default_current_hex is not None:
        if not isinstance(default_current_hex, str) or not default_current_hex.startswith("#"):
            raise ValueError("`assign_status_colors`: Invalid default_current_hex provided.")

    def assign_hex(row):
        if row["duration_status"] == "Current tracks":
            if use_hex_column_for_current:
                return row[hex_column]
            else:
                return default_current_hex if default_current_hex else previous_color_hex
        elif row["duration_status"] == "Previous tracks":
            return previous_color_hex
        else:
            return previous_color_hex

    df["duration_status_hex_colors"] = df.apply(assign_hex, axis=1)
    df["duration_status_colors"] = df["duration_status_hex_colors"].apply(hex_to_rgba)

    return df


@task
def assign_season_colors(gdf: AnyDataFrame, seasons_column: str) -> AnyDataFrame:
    """
    Assign hex and RGBA colors for seasons (wet/dry).

    Args:
        gdf: GeoDataFrame/DataFrame with season information.
        seasons_column: Column containing season values ('wet' or 'dry').

    Returns:
        GeoDataFrame with two new columns:
        - season_hex_colors: hex color codes
        - season_colors: RGBA tuples

    Note:
        If seasons_column is not present, creates it with 'undefined' value.
    """
    import pandas as pd

    seasons_dict = {
        "wet": "#255084",  # Blue for wet season
        "dry": "#f57c00",  # Orange for dry season
        "undefined": "#808080",  # Gray for undefined season
    }

    if gdf is None or gdf.empty:
        raise ValueError("`assign_season_colors`: gdf is empty.")

    df = gdf.copy()

    # Create season column with 'undefined' if it doesn't exist
    if seasons_column not in df.columns:
        df[seasons_column] = "undefined"

    # Validate that season values are recognized
    unique_seasons = df[seasons_column].dropna().unique()
    invalid_seasons = set(unique_seasons) - set(seasons_dict.keys())
    if invalid_seasons:
        raise ValueError(
            f"`assign_season_colors`: Invalid season values found: {invalid_seasons}. "
            f"Expected 'wet', 'dry', or 'undefined'."
        )

    # Reset index if MultiIndex to avoid issues with apply
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(drop=False)

    # Assign hex colors based on season
    df["season_hex_colors"] = df[seasons_column].map(seasons_dict)

    # Convert hex to RGBA tuples, handling potential None values
    df["season_colors"] = df["season_hex_colors"].apply(lambda x: hex_to_rgba(x) if pd.notna(x) else None)

    return df
