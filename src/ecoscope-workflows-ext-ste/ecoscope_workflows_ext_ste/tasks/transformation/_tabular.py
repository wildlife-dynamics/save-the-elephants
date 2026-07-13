import warnings
from pydantic import Field
from wt_registry import register
from collections.abc import Mapping
from ecoscope.platform.annotations import (  # type: ignore[import-untyped]
    AnyDataFrame,
)
from typing import List, Optional, cast, Any, Annotated, Union


@register()
def subset_columns(
    df: AnyDataFrame,
    columns: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    strict: bool = False,
) -> AnyDataFrame:
    """
    Return a DataFrame restricted to a subset of columns.

    Provide either `columns` (allowlist) or `exclude` (denylist), not both.
    Names not present in the DataFrame are skipped with a warning, or raise
    a KeyError if `strict=True`.

    Parameters
    ----------
    df : AnyDataFrame
        Input DataFrame.
    columns : list of str, optional
        Column names to keep, in the given order.
    exclude : list of str, optional
        Column names to drop; remaining columns keep their original order.
    strict : bool, default False
        If True, a missing column name raises KeyError instead of warning.

    Returns
    -------
    AnyDataFrame
        A new DataFrame with the selected columns.
    """
    if columns is not None and exclude is not None:
        raise ValueError("Pass either `columns` or `exclude`, not both.")

    existing = set(df.columns)

    if columns is not None:
        missing = [c for c in columns if c not in existing]
        if missing:
            if strict:
                raise KeyError(f"Columns not found: {missing}")
            warnings.warn(f"Columns not found, skipping: {missing}", stacklevel=2)
        keep = [c for c in columns if c in existing]
        print(
            f"[subset_columns] Keeping {len(keep)} of {len(df.columns)} columns from {len(df)}-row DataFrame"
            + (f" (skipped {len(missing)} missing: {missing})" if missing else "")
        )
        return df[keep].copy()

    if exclude is not None:
        missing = [c for c in exclude if c not in existing]
        if missing:
            if strict:
                raise KeyError(f"Columns not found: {missing}")
            warnings.warn(f"Columns not found, skipping exclusion: {missing}", stacklevel=2)
        drop = set(exclude)
        keep = [c for c in df.columns if c not in drop]
        print(f"[subset_columns] Dropped {len(drop)} columns, {len(keep)} remaining from {len(df)}-row DataFrame")
        return df[keep].copy()

    print(f"[subset_columns] No filter applied — returning all {len(df.columns)} columns as-is")
    return df.copy()


@register()
def add_mapped_column_value(
    df: AnyDataFrame,
    column: str,
    mapping: Mapping[Any, Any],
    new_column: str | None = None,
    default: Any = None,
    keep_unmapped: bool = False,
) -> AnyDataFrame:
    """
    Map values in ``column`` through ``mapping`` and write the result to a new column.

    Parameters
    ----------
    df : AnyDataFrame
        Input DataFrame.
    column : str
        Source column to read values from.
    mapping : Mapping[Any, Any]
        Lookup from source values to target values.
    new_column : str | None
        Destination column name. Defaults to ``f"{col}_mapped"``.
    default : Any
        Value to use for source values not present in ``mapping``.
        Ignored if ``keep_unmapped`` is True.
    keep_unmapped : bool
        If True, source values not in ``mapping`` are passed through unchanged.
        Mutually exclusive with a non-None ``default``.

    Returns
    -------
    AnyDataFrame
        Copy of ``df`` with the mapped column added.
    """
    dest = new_column or f"{column}_mapped"
    if column not in df.columns:
        raise KeyError(f"col {column!r} not in DataFrame columns")

    new_column = dest
    mapped = df[column].map(mapping)
    if keep_unmapped:
        mapped = mapped.fillna(df[column])
    elif default is not None:
        mapped = mapped.fillna(default)

    df[new_column] = mapped
    return cast(AnyDataFrame, df)


@register()
def add_new_column(df: AnyDataFrame, column_name: str, default_value: int | float | str) -> AnyDataFrame:
    """
    Create a new column in the DataFrame with a default value if it doesn't already exist.

    Args:
        df: Input DataFrame
        col_name: Name of the column to create
        default_value: Default value to assign to the new column

    Returns:
        DataFrame with the new column added (if it was missing)
    """
    df = df.copy()
    if column_name not in df.columns:
        df[column_name] = default_value
        print(f"[add_new_column] Added column '{column_name}' = {default_value!r} across {len(df)} rows")
    else:
        print(f"[add_new_column] Column '{column_name}' already exists — skipping")
    return df


@register()
def column_first_unique_value(
    df: AnyDataFrame,
    column_name: Annotated[str, Field(description="Column to aggregate")],
) -> Annotated[str, Field(description="The first unique string value in the column (sentence case)")]:
    """
    Get the first unique value from a column.

    Raises:
        ValueError: If df is empty or column doesn't exist.
    """
    if df is None or df.empty:
        raise ValueError("df is empty")

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    unique_values = df[column_name].unique()

    if len(unique_values) == 0:
        raise ValueError(f"No values found in column '{column_name}'")

    return str(unique_values[0])


@register()
def convert_columns_to_string(
    df: AnyDataFrame,
    columns: Union[str, List[str]],
) -> AnyDataFrame:
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in DataFrame. Skipping.")
            continue

        try:
            df[column] = df[column].astype(str)
        except Exception as e:
            print(f"Error converting column '{column}' to int: {e}")

    return df


@register()
def safe_string(
    value: Annotated[str, Field(description="String to make safe for use as a filename")],
) -> str:
    """Sanitize a string for filenames: replace spaces with underscores, remove special characters, lowercase."""
    import re

    safe = re.sub(r"[^\w\s-]", "", value)
    safe = re.sub(r"\s+", "_", safe)
    return safe.lower().strip("_")


@register()
def round_off_values(value: float, dp: int) -> float:
    return round(value, dp)
