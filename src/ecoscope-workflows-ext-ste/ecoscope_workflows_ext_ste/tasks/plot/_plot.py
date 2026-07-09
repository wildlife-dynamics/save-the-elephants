from pydantic import Field
from typing import Annotated
from wt_registry import register
from ecoscope.platform.annotations import (
    AdvancedField,
    DataFrame,
    JsonSerializableDataFrameModel,
)
from pydantic.json_schema import SkipJsonSchema
from ecoscope.platform.tasks.results._ecoplot import (
    BarLayoutStyle,
    ExportArgs,
    GroupedPlotStyle,
    PlotStyle,
)


@register()
def draw_stacked_percentage_bar_chart(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    group_by: Annotated[
        list[str],
        Field(
            description="The dataframe column(s) defining the bars. "
            "With two or more columns, a multicategory x-axis is used."
        ),
    ],
    category_column: Annotated[
        str,
        Field(description="The dataframe column whose values segment each bar."),
    ],
    grouped_styles: Annotated[
        list[GroupedPlotStyle] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Per-category styles. List order sets the stacking and "
            "legend order. If set, rows whose category is not listed are "
            "dropped before percentages are computed, so visible segments "
            "always sum to 100%. If unset, categories are discovered from the "
            "data and plotted in sorted order.",
        ),
    ] = None,
    color_column: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="The name of the dataframe column to color bars with. "
            "Each category should map to a single color value; an explicit "
            "marker_color in grouped_styles takes precedence.",
        ),
    ] = None,
    y_axis_title: Annotated[
        str,
        AdvancedField(default="% of rows", description="The y axis title."),
    ] = "% of rows",
    plot_style: Annotated[
        PlotStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Additional style kwargs passed to go.Bar() for all categories.",
        ),
    ] = None,
    layout_style: Annotated[
        BarLayoutStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Additional kwargs passed to plotly.go.Figure(layout). "
            "Overrides the task's layout defaults.",
        ),
    ] = None,
    widget_id: Annotated[
        str | SkipJsonSchema[None],
        Field(
            description="""\
            The id of the dashboard widget that this tile layer belongs to.
            If set this MUST match the widget title as defined downstream in create_widget tasks
            """,
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field()]:
    """
    Generates a stacked bar chart of within-group percentages for a categorical column

    Args:
    dataframe (pd.DataFrame): The input dataframe.
    group_by (list[str]): The dataframe column(s) defining the bars.
    category_column (str): The dataframe column whose values segment each bar.
    grouped_styles (list[GroupedPlotStyle]): Per-category styles; order sets stacking/legend order.
    color_column (str): The name of the dataframe column to color bars with.
    y_axis_title (str): The y axis title.
    plot_style (PlotStyle): Additional style kwargs passed to go.Bar() for all categories.
    layout_style (BarLayoutStyle): Additional kwargs passed to plotly.go.Figure(layout).
    widget_id (str): The id of the dashboard widget that this tile layer belongs to.
        If set this MUST match the widget title as defined downstream in create_widget tasks

    Returns:
    The generated chart html as a string
    """
    import logging

    import plotly.graph_objects as go  # type: ignore[import-untyped]

    logger = logging.getLogger(__name__)

    if dataframe is None or dataframe.empty:
        raise ValueError("dataframe is empty.")
    if not group_by:
        raise ValueError("group_by must contain at least one column.")

    required = [*group_by, category_column]
    if color_column:
        required.append(color_column)
    missing = [c for c in required if c not in dataframe.columns]
    if missing:
        raise ValueError(f"dataframe is missing required columns: {missing}.")

    if grouped_styles:
        styles_by_category = {s.category: s.plot_style for s in grouped_styles}
        unknown = set(dataframe[category_column].dropna().unique()) - set(styles_by_category)
        if unknown:
            logger.warning("Dropping unknown %s values: %s", category_column, sorted(unknown))
        relevant = dataframe[dataframe[category_column].isin(styles_by_category)]
        if relevant.empty:
            raise ValueError(f"No rows with known {category_column}. Found: {sorted(unknown)}")
    else:
        observed = sorted(dataframe[category_column].dropna().unique())
        styles_by_category = {cat: None for cat in observed}
        relevant = dataframe

    colors_by_category: dict = {}
    if color_column:
        color_counts = relevant.groupby(category_column)[color_column].nunique()
        ambiguous = color_counts[color_counts > 1]
        if not ambiguous.empty:
            logger.warning(
                "%s maps to multiple %s values for categories %s; using the first.",
                category_column,
                color_column,
                sorted(ambiguous.index),
            )
        colors_by_category = relevant.groupby(category_column)[color_column].first().to_dict()

    counts = relevant.groupby([*group_by, category_column]).size().reset_index(name="count")
    counts["total"] = counts.groupby(group_by)["count"].transform("sum")
    counts["pct"] = counts["count"] / counts["total"] * 100

    base_bar_kwargs = plot_style.model_dump(exclude_none=True) if plot_style else {}

    fig = go.Figure()
    for category, category_style in styles_by_category.items():
        plot_df = counts[counts[category_column] == category]
        if plot_df.empty:
            continue
        x = plot_df[group_by[0]] if len(group_by) == 1 else [plot_df[c] for c in group_by]
        category_kwargs = category_style.model_dump(exclude_none=True) if category_style else {}
        if "marker_color" not in category_kwargs and category in colors_by_category:
            category_kwargs["marker_color"] = colors_by_category[category]
        fig.add_trace(
            go.Bar(
                x=x,
                y=plot_df["pct"],
                name=str(category),
                hovertemplate=(
                    f"<b>%{{x}}</b><br>{category_column}: {category}<br>" "Percentage: %{y:.1f}%<br><extra></extra>"
                ),
                **base_bar_kwargs,
                **category_kwargs,
            )
        )

    xaxis = dict(
        tickfont=dict(size=12),
        title_standoff=20,
        constraintoward="left",
        autotickangles=[0, 90],
        ticklabeloverflow="hide past domain",
        ticklabelposition="outside",
    )
    if len(group_by) > 1:
        xaxis.update(
            type="multicategory",
            showdividers=True,
            dividercolor="grey",
            dividerwidth=1,
        )

    fig.update_layout(
        template="simple_white",
        barmode="stack",
        xaxis=xaxis,
        yaxis=dict(title_text=y_axis_title, ticksuffix="%", range=[0, 100], dtick=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.05,
        bargroupgap=0.05,
    )
    # User overrides win over the defaults above; plotly merges nested dicts.
    if layout_style:
        fig.update_layout(**layout_style.model_dump(exclude_none=True))

    return fig.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))
