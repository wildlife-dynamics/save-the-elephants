import logging
from ecoscope.base.utils import hex_to_rgba
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyDataFrame


logger = logging.getLogger(__name__)


@task
def convert_hex_to_rgba(df: AnyDataFrame, col: str, new_col: str) -> AnyDataFrame:
    """
    Converts a column of hex colors into RGBA format and stores the result in a specified column.

    Args:
        df (AnyDataFrame): Input DataFrame containing the color column.
        col (str): Name of the column containing hex color strings.
        new_col (str): Name of the new column to store RGBA values.

    Returns:
        AnyDataFrame: DataFrame with an added column containing RGBA tuples.
    """
    df[new_col] = df[col].apply(lambda hex_val: hex_to_rgba(hex_val))
    return df
