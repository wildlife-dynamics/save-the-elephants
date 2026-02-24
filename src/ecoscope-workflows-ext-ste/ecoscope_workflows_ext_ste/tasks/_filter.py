import logging
from pydantic import Field
from typing import Literal
from typing import Annotated, cast, Union
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyDataFrame
from ecoscope_workflows_core.tasks.transformation._filter import ComparisonOperator

logger = logging.getLogger(__name__)


@task
def filter_groups_by_value_criteria(
    df: Annotated[AnyDataFrame, Field(description="DataFrame to filter")],
    groupby_col: Annotated[str, Field(description="Column to group by")],
    filter_col: Annotated[str, Field(description="Column to check values in")],
    criteria: Annotated[
        Literal["all", "any", "none", "exactly", "priority"], Field(description="Filtering strategy")
    ] = "all",
    required_values: Annotated[
        list[str] | None, Field(description="Values to check for (required for 'all', 'any', 'exactly' criteria)")
    ] = None,
    priority_value: Annotated[str | None, Field(description="Must-have value for 'priority' criteria")] = None,
    min_count: Annotated[
        int | None, Field(description="Minimum number of unique values required in filter_col per group")
    ] = None,
) -> AnyDataFrame:
    """
    Filter DataFrame groups based on various value criteria.

    Strategies:
    - 'all': Keep groups that have ALL required_values
    - 'any': Keep groups that have ANY of required_values
    - 'none': Keep groups that have NONE of required_values
    - 'exactly': Keep groups that have EXACTLY the required_values (no more, no less)
    - 'priority': Keep groups that have the priority_value (must-have), regardless of other values

    Args:
        df: Input DataFrame
        groupby_col: Column to group by
        filter_col: Column to check values in
        criteria: Filtering strategy
        required_values: List of values to check for
        priority_value: Must-have value for 'priority' criteria
        min_count: Minimum number of unique values in filter_col (optional)

    Examples:
        # Keep only subjects with both current and previous tracks
        filter_groups_by_value_criteria(
            df,
            groupby_col='subject_name',
            filter_col='duration_status',
            criteria='all',
            required_values=['Current tracks', 'Previous tracks']
        )

        # Keep subjects with at least one track type
        filter_groups_by_value_criteria(
            df,
            groupby_col='subject_name',
            filter_col='duration_status',
            criteria='any',
            required_values=['Current tracks', 'Previous tracks']
        )

        # Keep subjects that MUST have current tracks (may or may not have previous)
        filter_groups_by_value_criteria(
            df,
            groupby_col='subject_name',
            filter_col='duration_status',
            criteria='priority',
            priority_value='Current tracks'
        )

        # Keep subjects with at least 10 observations
        filter_groups_by_value_criteria(
            df,
            groupby_col='subject_name',
            filter_col='fixtime',
            min_count=10
        )
    """
    logger.info(f"Starting filter_groups_by_value_criteria with criteria='{criteria}'")
    if df is None or df.empty:
        raise ValueError("filter_groups_by_value_criteria: Input DataFrame is empty")

    if groupby_col not in df.columns:
        logger.info(f"Column '{groupby_col}' not found in DataFrame. Returning original DataFrame.")
        return df

    if filter_col not in df.columns:
        raise ValueError(f"filter_groups_by_value_criteria: Column '{filter_col}' not found")

    if criteria in ["all", "any", "none", "exactly"] and not required_values:
        raise ValueError(f"filter_groups_by_value_criteria: required_values needed for '{criteria}' criteria")

    if criteria == "priority" and not priority_value:
        raise ValueError("filter_groups_by_value_criteria: priority_value needed for 'priority' criteria")

    logger.info(f"Filtering groups by {groupby_col} using criteria '{criteria}' on {filter_col}")

    # Get unique values in filter_col for each group
    group_values = df.groupby(groupby_col)[filter_col].apply(lambda x: set(x.unique()))
    group_counts = df.groupby(groupby_col)[filter_col].count()

    # Apply filtering criteria
    required_set = set(required_values) if required_values else set()

    if criteria == "all":
        # Keep groups that have ALL required values
        keep_mask = group_values.apply(lambda x: required_set.issubset(x))

    elif criteria == "any":
        # Keep groups that have ANY of the required values
        keep_mask = group_values.apply(lambda x: bool(required_set & x))

    elif criteria == "none":
        # Keep groups that have NONE of the required values
        keep_mask = group_values.apply(lambda x: not bool(required_set & x))

    elif criteria == "exactly":
        # Keep groups that have EXACTLY the required values (no more, no less)
        keep_mask = group_values.apply(lambda x: x == required_set)

    elif criteria == "priority":
        # Keep groups that have the priority value (must-have)
        keep_mask = group_values.apply(lambda x: priority_value in x)
        logger.info(f"Applying priority filter: groups must contain '{priority_value}'")

    else:
        raise ValueError(f"Unknown criteria: {criteria}")

    # Apply min_count filter if specified
    if min_count is not None:
        count_mask = group_counts >= min_count
        keep_mask = keep_mask & count_mask
        logger.info(f"Also applying min_count filter: {min_count}")

    complete_groups = group_values[keep_mask]

    logger.info(f"Groups before filtering: {df[groupby_col].nunique()}")
    logger.info(f"Groups matching criteria: {len(complete_groups)}")

    if len(complete_groups) == 0:
        logger.info("WARNING: No groups match the criteria")
        logger.info("Value combinations per group:")
        for group_name, values in group_values.items():
            count = group_counts[group_name]
            logger.info(f"  {group_name}: {values} (n={count})")
        return df.iloc[:0].copy()

    # Filter DataFrame
    filtered_df = df[df[groupby_col].isin(complete_groups.index)].copy()

    # Logging
    all_groups = set(df[groupby_col].unique())
    kept_groups = set(complete_groups.index)
    dropped_groups = all_groups - kept_groups

    if kept_groups:
        logger.info(f"Kept groups ({len(kept_groups)}): {sorted(kept_groups)}")
    if dropped_groups:
        logger.info(f"Dropped groups ({len(dropped_groups)}): {sorted(dropped_groups)}")
        for group in sorted(dropped_groups):
            group_vals = group_values[group]
            count = group_counts[group]
            logger.info(f"  {group}: {group_vals} (n={count})")

    logger.info(f"Rows: {len(df)} -> {len(filtered_df)}")

    return filtered_df


@task
def filter_df_values(
    df: Annotated[
        AnyDataFrame,
        Field(
            description="The dataframe.",
            exclude=True,
        ),
    ],
    column_name: Annotated[str, Field(description="The column name to filter on.")],
    op: Annotated[ComparisonOperator, Field(description="The comparison operator")],
    value: Annotated[Union[float, int, str], Field(description="The comparison operand (numeric or string)")],
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
