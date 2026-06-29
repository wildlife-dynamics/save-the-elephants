import warnings
from wt_registry import register
from typing import List, Optional
from ecoscope.platform.annotations import (  # type: ignore[import-untyped]
    AnyDataFrame,
)


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
        return df[keep].copy()

    if exclude is not None:
        missing = [c for c in exclude if c not in existing]
        if missing:
            if strict:
                raise KeyError(f"Columns not found: {missing}")
            warnings.warn(f"Columns not found, skipping exclusion: {missing}", stacklevel=2)
        drop = set(exclude)
        keep = [c for c in df.columns if c not in drop]
        return df[keep].copy()

    return df.copy()
