import pandas as pd
from typing import List
from wt_registry import register
from typing import Literal, Optional, Sequence
from ecoscope.platform.annotations import AnyDataFrame


@register()
def concatenate_dataframes(
    list_df: List[AnyDataFrame],
    axis: Literal[0, 1, "index", "columns"] = 0,
    join: Literal["inner", "outer"] = "outer",
    ignore_index: bool = True,
    keys: Optional[Sequence] = None,
    levels: Optional[Sequence] = None,
    names: Optional[Sequence[str]] = None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: Optional[bool] = None,
) -> AnyDataFrame:
    """
    Merge multiple dataframes into a single dataframe.

    Args mirror pandas.concat:
        list_df: Dataframes to concatenate.
        axis: 0/"index" to stack rows, 1/"columns" to join side by side.
        join: How to handle the other axis — "outer" (union) or "inner"
            (intersection).
        ignore_index: If True, don't preserve index values along the
            concatenation axis (defaulted True, unlike pandas).
        keys: Build a hierarchical index using these as the outermost level.
        levels: Specific levels to use for the resulting MultiIndex.
        names: Names for the levels in the resulting MultiIndex.
        verify_integrity: If True, raise on duplicate values in the new axis.
        sort: Sort the non-concatenation axis if not already aligned.
        copy: Passed through to pandas (deprecated/no-op in pandas 3.x).

    Returns:
        A single merged dataframe.
    """
    if not list_df:
        raise ValueError("list_df cannot be empty")

    kwargs = dict(
        axis=axis,
        join=join,
        ignore_index=ignore_index,
        keys=keys,
        levels=levels,
        names=names,
        verify_integrity=verify_integrity,
        sort=sort,
    )
    if copy is not None:  # avoid triggering the pandas 3.x deprecation warning
        kwargs["copy"] = copy

    return pd.concat(list_df, **kwargs)
