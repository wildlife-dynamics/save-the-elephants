from wt_registry import register
import plotly.graph_objects as go  # type: ignore[import-untyped
from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.tasks.results._ecoplot import ExportArgs


@register()
def plot_fix_protection_status(gdf: AnyDataFrame) -> str:
    """
    Stacked bar chart of fix percentages by protection status, grouped by
    subject and season.

    Each bar represents one (subject, season) combination, segmented into
    Protected and Unprotected portions. Rows with `protection_status` not
    in {Protected, Unprotected} are dropped with a warning.

    Args:
        gdf: DataFrame with columns: `subject_name`, `season`,
            `protection_status`.

    Returns:
        HTML string of the rendered Plotly figure.

    Raises:
        ValueError: If `gdf` is empty or missing required columns.
    """
    if gdf is None or gdf.empty:
        raise ValueError("gdf is empty.")

    required = ["subject_name", "season", "protection_status"]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"gdf is missing required columns: {missing}.")

    colors = {"Protected": "#006400", "Unprotected": "#ff8c00"}

    # Drop unknown statuses before percentage computation so visible bars sum to 100%.
    unknown = set(gdf["protection_status"].dropna().unique()) - colors.keys()
    if unknown:
        print("Dropping unknown protection_status values: %s", unknown)

    relevant = gdf[gdf["protection_status"].isin(colors.keys())]
    if relevant.empty:
        raise ValueError(f"No rows with known protection_status. Found: {sorted(unknown)}")

    fix_counts = relevant.groupby(["subject_name", "season", "protection_status"]).size().reset_index(name="fix_count")
    fix_counts["total"] = fix_counts.groupby(["subject_name", "season"])["fix_count"].transform("sum")
    fix_counts["fix_pct"] = (fix_counts["fix_count"] / fix_counts["total"]) * 100

    fig = go.Figure()
    for status, color in colors.items():
        plot_df = fix_counts[fix_counts["protection_status"] == status]
        if plot_df.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=[plot_df["subject_name"], plot_df["season"]],
                y=plot_df["fix_pct"],
                name=status,
                marker_color=color,
                hovertemplate=(f"<b>%{{x}}</b><br>Status: {status}<br>" "Percentage: %{y:.1f}%<br><extra></extra>"),
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
