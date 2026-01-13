import pandas as pd
from typing import List
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyDataFrame


@task
def merge_multiple_df(list_df: List[AnyDataFrame], ignore_index: bool = True, sort: bool = False) -> AnyDataFrame:
    """
    Merge multiple dataframes into a single dataframe.

    Args:
        list_df: List of dataframes to concatenate
        ignore_index: If True, do not use the index values along the concatenation axis
        sort: Sort non-concatenation axis if it is not already aligned

    Returns:
        A single merged dataframe
    """
    if not list_df:
        raise ValueError("list_df cannot be empty")

    merged_df = pd.concat(list_df, ignore_index=ignore_index, sort=sort)
    return merged_df
