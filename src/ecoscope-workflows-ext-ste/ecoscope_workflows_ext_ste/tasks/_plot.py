import plotly.graph_objects as go
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot import ExportArgs
from ecoscope_workflows_core.annotations import AnyDataFrame
from typing import Annotated
from pydantic import Field


@task
def plot_fix_protection_status(gdf: AnyDataFrame) -> Annotated[str, Field()]:
    """
    Plot the percentage of fixes by protection status (Protected/Unprotected)
    grouped by subject name and season.

    Args:
        gdf: DataFrame containing columns: subject_name, season, protection_status

    Returns:
        HTML string of the rendered Plotly figure.
    """
    fix_counts = gdf.groupby(["subject_name", "season", "protection_status"]).size().reset_index(name="fix_count")

    fix_counts["total"] = fix_counts.groupby(["subject_name", "season"])["fix_count"].transform("sum")
    fix_counts["fix_pct"] = (fix_counts["fix_count"] / fix_counts["total"]) * 100

    fig = go.Figure()

    colors = {"Protected": "#006400", "Unprotected": "#ff8c00"}

    for status, color in colors.items():
        plot_df = fix_counts[fix_counts["protection_status"] == status]
        fig.add_trace(
            go.Bar(
                x=[plot_df["subject_name"], plot_df["season"]],
                y=plot_df["fix_pct"],
                name=status,
                marker_color=color,
                hovertemplate=("<b>%{x}</b><br>" f"Status: {status}<br>" "Percentage: %{y:.1f}%<br>" "<extra></extra>"),
            )
        )

    fig.update_layout(
        template="simple_white",
        barmode="stack",
        xaxis=dict(
            tickfont=dict(size=12),
            title_standoff=20,
            showdividers=True,
            dividercolor="grey",
            dividerwidth=1,
            constraintoward="left",
            autotickangles=[0, 90],
            type="multicategory",
            ticklabeloverflow="hide past domain",
            ticklabelposition="outside",
        ),
        yaxis=dict(
            title_text="% of Fixes",
            ticksuffix="%",
            range=[0, 100],
            dtick=10,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        bargap=0.05,
        bargroupgap=0.05,
    )

    return fig.to_html(**ExportArgs().model_dump(exclude_none=True))
