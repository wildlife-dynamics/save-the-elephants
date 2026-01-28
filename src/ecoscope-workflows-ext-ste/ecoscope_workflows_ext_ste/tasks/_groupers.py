import logging
from pydantic import Field
from typing import Annotated, Any, Union, List
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.indexes import CompositeFilter
from pydantic.json_schema import SkipJsonSchema

from ecoscope_workflows_core.indexes import (
    AllGrouper,
    TemporalGrouper,
    ValueAndOrTemporalGroupersBeforeValidator,
    ValueGrouper,
)
from ecoscope_workflows_core.tasks.groupby._groupby import _groupers_field_json_schema_extra

logger = logging.getLogger(__name__)


@task
def get_split_group_column(
    split_data: Annotated[
        list[tuple[CompositeFilter, Any]],
        Field(description="Output from split_groups: [(CompositeFilter, df), ...]"),
    ],
) -> str | None:
    """
    Extract the column name used for the first split.

    Example:
        Input: [
            ((('region', '=', 'Atlantida'), ('year', '=', '2024')), df1),
            ((('region', '=', 'Yoro'),), df2),
        ]
        Output: 'region'
    """
    if not split_data:
        return None

    # Get the first composite filter
    composite_filter, _ = split_data[0]

    if composite_filter:
        # Extract column name from the first filter tuple
        column_name, _, _ = composite_filter[0]
        return column_name
    logging.info(f"column name: {column_name}")
    return None


@task
def get_split_group_values(
    split_data: Annotated[
        list[tuple[CompositeFilter, Any]],
        Field(description="Output from split_groups: [(CompositeFilter, df), ...]"),
    ],
) -> list[dict[str, Any]]:
    """
    Extract all grouper values from each split group as dictionaries.

    Example:
        Input: [
            ((('region', '=', 'Atlantida'), ('year', '=', '2024')), df1),
            ((('region', '=', 'Yoro'),), df2),
        ]
        Output: [{'region': 'Atlantida', 'year': '2024'}, {'region': 'Yoro'}]
    """
    values_list: list[dict[str, Any]] = []
    for composite_filter, _ in split_data:
        group_values: dict[str, Any] = {}
        for index_name, _, value in composite_filter:
            group_values[index_name] = value
        values_list.append(group_values)
        logging.info(f"values list: {values_list}")
    return values_list


@task
def get_split_group_names(
    split_data: Annotated[
        list[tuple[CompositeFilter, Any]],
        Field(description="Output from split_groups: [(CompositeFilter, df), ...]"),
    ],
) -> list[str]:
    """
    Extract the first grouper value from each split group.

    Example:
        Input: [
            ((('region', '=', 'Atlantida'), ('year', '=', '2024')), df1),
            ((('region', '=', 'Yoro'),), df2),
        ]
        Output: ['Atlantida', 'Yoro']
    """
    names: list[str] = []
    for composite_filter, _ in split_data:
        if composite_filter:
            # take the first filter tuple: (index_name, '=', value)
            _, _, value = composite_filter[0]
            names.append(str(value))
        else:
            names.append("Unknown")
    return names


@task
def extract_index_names(
    groupers: Union[List[Union[ValueGrouper, AllGrouper, TemporalGrouper]], ValueGrouper, AllGrouper, TemporalGrouper],
) -> str:
    """
    Extract index_name values from ValueGrouper, AllGrouper, or TemporalGrouper objects.

    Args:
        groupers: Can be a single grouper object or a list of grouper objects

    Returns:
        string of index names (comma-separated if multiple), lowercase
        Returns 'all' for AllGrouper
        Returns the temporal directive for TemporalGrouper (the actual column name)
    """
    # Handle single object input
    if not isinstance(groupers, list):
        groupers = [groupers]

    # Check if any grouper is AllGrouper - return 'all'
    if any(isinstance(g, AllGrouper) for g in groupers):
        return "all"

    # Build list of index names
    index_names = []
    for grouper in groupers:
        if isinstance(grouper, TemporalGrouper):
            # For TemporalGrouper, use the directive as the column name (lowercase)
            index_names.append(grouper.temporal_index.directive.lower())
        else:  # ValueGrouper
            index_names.append(grouper.index_name.lower())

    index_name_str = index_names[0]
    return index_name_str


@task
def set_custom_groupers(
    groupers: Annotated[
        list[ValueGrouper] | SkipJsonSchema[None],
        Field(
            default=None,
            title=" ",  # deliberately a single empty space, to hide the field in the UI
            json_schema_extra=_groupers_field_json_schema_extra,
            description="""\
            Specify how the data should be grouped to create the views for your dashboard.
            This field is optional; if left blank, all the data will appear in a single view.
            """,
        ),
        ValueAndOrTemporalGroupersBeforeValidator,
    ] = None,
) -> Annotated[
    AllGrouper | list[ValueGrouper],
    Field(
        description="""\
        Passthrough of the input groupers, for use in downstream tasks.
        If no groupers are given, the `AllGrouper` is returned instead.
        """,
    ),
]:
    return groupers if groupers else AllGrouper()
