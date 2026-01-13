from pydantic import Field
from typing import Literal
from typing import Annotated
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyDataFrame


@task
def filter_groups_by_value_criteria(
    df: Annotated[AnyDataFrame, Field(description="DataFrame to filter")],
    groupby_col: Annotated[str, Field(description="Column to group by")],
    filter_col: Annotated[str, Field(description="Column to check values in")],
    criteria: Annotated[Literal["all", "any", "none", "exactly"], Field(description="Filtering strategy")] = "all",
    required_values: Annotated[
        list[str] | None, Field(description="Values to check for (required for 'all', 'any', 'exactly' criteria)")
    ] = None,
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

    Args:
        df: Input DataFrame
        groupby_col: Column to group by
        filter_col: Column to check values in
        criteria: Filtering strategy
        required_values: List of values to check for
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

        # Keep subjects with at least 10 observations
        filter_groups_by_value_criteria(
            df,
            groupby_col='subject_name',
            filter_col='fixtime',
            min_count=10
        )
    """

    if df is None or df.empty:
        raise ValueError("filter_groups_by_value_criteria: Input DataFrame is empty")

    if groupby_col not in df.columns:
        print(f"Column '{groupby_col}' not found in DataFrame. Returning original DataFrame.")
        return df

    if filter_col not in df.columns:
        raise ValueError(f"filter_groups_by_value_criteria: Column '{filter_col}' not found")

    if criteria in ["all", "any", "none", "exactly"] and not required_values:
        raise ValueError(f"filter_groups_by_value_criteria: required_values needed for '{criteria}' criteria")

    print(f"Filtering groups by {groupby_col} using criteria '{criteria}' on {filter_col}")

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

    else:
        raise ValueError(f"Unknown criteria: {criteria}")

    # Apply min_count filter if specified
    if min_count is not None:
        count_mask = group_counts >= min_count
        keep_mask = keep_mask & count_mask
        print(f"Also applying min_count filter: {min_count}")

    complete_groups = group_values[keep_mask]

    print(f"Groups before filtering: {df[groupby_col].nunique()}")
    print(f"Groups matching criteria: {len(complete_groups)}")

    if len(complete_groups) == 0:
        print("WARNING: No groups match the criteria")
        print("Value combinations per group:")
        for group_name, values in group_values.items():
            count = group_counts[group_name]
            print(f"  {group_name}: {values} (n={count})")
        return df.iloc[:0].copy()

    # Filter DataFrame
    filtered_df = df[df[groupby_col].isin(complete_groups.index)].copy()

    # Logging
    all_groups = set(df[groupby_col].unique())
    kept_groups = set(complete_groups.index)
    dropped_groups = all_groups - kept_groups

    if kept_groups:
        print(f"Kept groups ({len(kept_groups)}): {sorted(kept_groups)}")
    if dropped_groups:
        print(f"Dropped groups ({len(dropped_groups)}): {sorted(dropped_groups)}")
        for group in sorted(dropped_groups):
            group_vals = group_values[group]
            count = group_counts[group]
            print(f"  {group}: {group_vals} (n={count})")

    print(f"Rows: {len(df)} → {len(filtered_df)}")

    return filtered_df
