from pydantic import Field
from wt_registry import register
from typing import Annotated, cast, Union
from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.tasks.transformation._filter import ComparisonOperator


@register()
def filter_rows(
    df: Annotated[
        AnyDataFrame,
        Field(
            description="The dataframe.",
            exclude=True,
        ),
    ],
    column_name: Annotated[str, Field(description="The column name to filter on.")],
    op: Annotated[ComparisonOperator, Field(description="The comparison operator")],
    value: Annotated[Union[bool, float, int, str], Field(description="The comparison operand (numeric or string)")],
    reset_index: Annotated[bool, Field(description="If reset index, default is False")] = False,
) -> AnyDataFrame:
    match op:
        case ComparisonOperator.EQUAL:
            result_df = df[df[column_name] == value]
        case ComparisonOperator.NE:
            result_df = df[df[column_name] != value]
        case ComparisonOperator.GE:
            result_df = df[df[column_name] >= value]
        case ComparisonOperator.GT:
            result_df = df[df[column_name] > value]
        case ComparisonOperator.LE:
            result_df = df[df[column_name] <= value]
        case ComparisonOperator.LT:
            result_df = df[df[column_name] < value]

    if reset_index:
        result_df = result_df.reset_index(drop=True)

    return cast(AnyDataFrame, result_df)
