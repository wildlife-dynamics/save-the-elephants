import geopandas as gpd
from typing import Union,cast,Sequence
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyDataFrame

@task
def filter_by_value(
    df: AnyDataFrame,
    column_name: str,
    value: Union[int, str, float, Sequence[Union[int, str, float]]]
) -> AnyDataFrame:
    """
    Return a DataFrame containing rows where the given column matches the specified value
    or any value in a list/sequence.
    """

    if isinstance(value, (list, tuple, set)):
        df_filtered = df[df[column_name].isin(value)].copy()
    else:
        df_filtered = df[df[column_name] == value].copy()

    return cast(AnyDataFrame, df_filtered)

@task
def exclude_by_value(
    df: AnyDataFrame,
    column_name: str,
    value: Union[int, str, float, Sequence[Union[int, str, float]]]
) -> AnyDataFrame:
    """
    Return a DataFrame containing rows where the given column matches the specified value
    or any value in a list/sequence.
    """

    if isinstance(value, (list, tuple, set)):
        df_filtered = df[~df[column_name].isin(value)].copy()
    else:
        df_filtered = df[df[column_name] != value].copy()

    return cast(AnyDataFrame, df_filtered)

