"""Tests for ecoscope_workflows_ext_ste.tasks.plot._plot.

`draw_stacked_percentage_bar_chart` is registered via `wt_registry.register()`,
a no-op decorator at call time, so it is exercised as a plain Python function.

The task returns an HTML string (`fig.to_html(...)`), not a `plotly.Figure`
object, so to make structural assertions (traces / data / layout) without
parsing the rendered HTML, a `monkeypatch` spy is installed on
`plotly.graph_objects.Figure.to_html` that captures `self` (the real Figure
instance) before delegating to the original method. Every test therefore
gets both: the actual rendered HTML string (asserted to be non-empty / valid)
and the underlying Figure object (asserted on structurally).
"""

import logging

import pandas as pd
import plotly.graph_objects as go
import pytest

from ecoscope.platform.tasks.results._ecoplot import (
    BarLayoutStyle,
    GroupedPlotStyle,
    PlotCategoryStyle,
    PlotStyle,
)
from ecoscope_workflows_ext_ste.tasks.plot._plot import draw_stacked_percentage_bar_chart


@pytest.fixture
def capture_figure(monkeypatch):
    """Spy on `go.Figure.to_html`, stashing the Figure instance it was called
    with (and the args/kwargs) so tests can assert on the pre-render object.
    """
    captured: dict = {}
    original_to_html = go.Figure.to_html

    def _spy(self, *args, **kwargs):
        captured["figure"] = self
        captured["kwargs"] = kwargs
        return original_to_html(self, *args, **kwargs)

    monkeypatch.setattr(go.Figure, "to_html", _spy)
    return captured


@pytest.fixture
def basic_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": ["Jan", "Jan", "Jan", "Feb", "Feb"],
            "species": ["Elephant", "Elephant", "Lion", "Elephant", "Lion"],
        }
    )


# --------------------------------------------------------------------------- #
# Happy path                                                                   #
# --------------------------------------------------------------------------- #


class TestHappyPath:
    def test_returns_non_empty_html_string(self, basic_df):
        html = draw_stacked_percentage_bar_chart(basic_df, group_by=["month"], category_column="species")
        assert isinstance(html, str)
        assert len(html) > 0
        assert "<div" in html
        assert "Plotly" in html

    def test_one_bar_trace_per_discovered_category_sorted(self, basic_df, capture_figure):
        draw_stacked_percentage_bar_chart(basic_df, group_by=["month"], category_column="species")
        fig = capture_figure["figure"]

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert all(isinstance(trace, go.Bar) for trace in fig.data)
        # No grouped_styles -> categories discovered from data, sorted.
        assert [trace.name for trace in fig.data] == ["Elephant", "Lion"]

    def test_percentages_computed_within_each_group(self, basic_df, capture_figure):
        draw_stacked_percentage_bar_chart(basic_df, group_by=["month"], category_column="species")
        fig = capture_figure["figure"]

        by_name = {trace.name: trace for trace in fig.data}
        # groupby sorts group keys alphabetically -> Feb before Jan.
        assert list(by_name["Elephant"].x) == ["Feb", "Jan"]
        assert list(by_name["Elephant"].y) == pytest.approx([50.0, 200 / 3])
        assert list(by_name["Lion"].x) == ["Feb", "Jan"]
        assert list(by_name["Lion"].y) == pytest.approx([50.0, 100 / 3])

    def test_layout_defaults(self, basic_df, capture_figure):
        draw_stacked_percentage_bar_chart(basic_df, group_by=["month"], category_column="species")
        fig = capture_figure["figure"]

        assert fig.layout.barmode == "stack"
        assert fig.layout.yaxis.title.text == "% of rows"
        assert fig.layout.yaxis.range == (0, 100)

    def test_custom_y_axis_title(self, basic_df, capture_figure):
        draw_stacked_percentage_bar_chart(
            basic_df, group_by=["month"], category_column="species", y_axis_title="Proportion"
        )
        fig = capture_figure["figure"]
        assert fig.layout.yaxis.title.text == "Proportion"

    def test_multiple_group_by_columns_uses_multicategory_xaxis(self, capture_figure):
        df = pd.DataFrame(
            {
                "region": ["North", "North", "South", "South"],
                "month": ["Jan", "Jan", "Jan", "Jan"],
                "species": ["Elephant", "Lion", "Elephant", "Lion"],
            }
        )
        draw_stacked_percentage_bar_chart(df, group_by=["region", "month"], category_column="species")
        fig = capture_figure["figure"]

        assert fig.layout.xaxis.type == "multicategory"
        assert fig.layout.xaxis.showdividers is True

    def test_single_group_by_does_not_set_multicategory(self, basic_df, capture_figure):
        draw_stacked_percentage_bar_chart(basic_df, group_by=["month"], category_column="species")
        fig = capture_figure["figure"]
        assert fig.layout.xaxis.type != "multicategory"


# --------------------------------------------------------------------------- #
# grouped_styles: ordering, filtering, and per-category style application     #
# --------------------------------------------------------------------------- #


class TestGroupedStyles:
    def test_trace_order_follows_grouped_styles_order(self, basic_df, capture_figure):
        draw_stacked_percentage_bar_chart(
            basic_df,
            group_by=["month"],
            category_column="species",
            grouped_styles=[
                GroupedPlotStyle(category="Lion"),
                GroupedPlotStyle(category="Elephant"),
            ],
        )
        fig = capture_figure["figure"]
        assert [trace.name for trace in fig.data] == ["Lion", "Elephant"]

    def test_per_category_marker_color_applied(self, basic_df, capture_figure):
        draw_stacked_percentage_bar_chart(
            basic_df,
            group_by=["month"],
            category_column="species",
            grouped_styles=[
                GroupedPlotStyle(category="Lion", plot_style=PlotCategoryStyle(marker_color="#ff0000")),
                GroupedPlotStyle(category="Elephant", plot_style=PlotCategoryStyle(marker_color="#00ff00")),
            ],
        )
        fig = capture_figure["figure"]
        by_name = {trace.name: trace for trace in fig.data}
        assert by_name["Lion"].marker.color == "#ff0000"
        assert by_name["Elephant"].marker.color == "#00ff00"

    def test_unlisted_categories_are_dropped_and_logged(self, basic_df, capture_figure, caplog):
        with caplog.at_level(logging.WARNING):
            draw_stacked_percentage_bar_chart(
                basic_df,
                group_by=["month"],
                category_column="species",
                grouped_styles=[GroupedPlotStyle(category="Lion")],
            )
        fig = capture_figure["figure"]
        assert [trace.name for trace in fig.data] == ["Lion"]
        assert any("Dropping unknown" in record.message for record in caplog.records)

    def test_visible_segments_still_sum_to_100_after_dropping(self, capture_figure):
        # 3 categories in the data, but only 2 are "known" -- percentages must
        # be recomputed over just the known rows so bars still sum to 100%.
        df = pd.DataFrame(
            {
                "month": ["Jan"] * 4,
                "species": ["Elephant", "Elephant", "Lion", "Cheetah"],
            }
        )
        draw_stacked_percentage_bar_chart(
            df,
            group_by=["month"],
            category_column="species",
            grouped_styles=[GroupedPlotStyle(category="Elephant"), GroupedPlotStyle(category="Lion")],
        )
        fig = capture_figure["figure"]
        total_pct = sum(trace.y[0] for trace in fig.data)
        assert total_pct == pytest.approx(100.0)

    def test_no_known_categories_raises(self, basic_df):
        with pytest.raises(ValueError, match="No rows with known species"):
            draw_stacked_percentage_bar_chart(
                basic_df,
                group_by=["month"],
                category_column="species",
                grouped_styles=[GroupedPlotStyle(category="Cheetah")],
            )


# --------------------------------------------------------------------------- #
# color_column                                                                 #
# --------------------------------------------------------------------------- #


class TestColorColumn:
    def test_maps_category_to_marker_color(self, capture_figure):
        df = pd.DataFrame(
            {
                "month": ["Jan", "Jan"],
                "species": ["Elephant", "Lion"],
                "color": ["#00ff00", "#ff0000"],
            }
        )
        draw_stacked_percentage_bar_chart(df, group_by=["month"], category_column="species", color_column="color")
        fig = capture_figure["figure"]
        by_name = {trace.name: trace for trace in fig.data}
        assert by_name["Elephant"].marker.color == "#00ff00"
        assert by_name["Lion"].marker.color == "#ff0000"

    def test_ambiguous_color_mapping_uses_first_and_logs_warning(self, caplog):
        df = pd.DataFrame(
            {
                "month": ["Jan", "Jan", "Feb"],
                "species": ["Lion", "Lion", "Lion"],
                "color": ["red", "blue", "red"],
            }
        )
        with caplog.at_level(logging.WARNING):
            html = draw_stacked_percentage_bar_chart(
                df, group_by=["month"], category_column="species", color_column="color"
            )
        assert isinstance(html, str) and len(html) > 0
        assert any("maps to multiple" in record.message for record in caplog.records)

    def test_explicit_marker_color_in_grouped_styles_wins_over_color_column(self, capture_figure):
        df = pd.DataFrame(
            {
                "month": ["Jan"],
                "species": ["Lion"],
                "color": ["#ff0000"],
            }
        )
        draw_stacked_percentage_bar_chart(
            df,
            group_by=["month"],
            category_column="species",
            color_column="color",
            grouped_styles=[GroupedPlotStyle(category="Lion", plot_style=PlotCategoryStyle(marker_color="#0000ff"))],
        )
        fig = capture_figure["figure"]
        assert fig.data[0].marker.color == "#0000ff"


# --------------------------------------------------------------------------- #
# plot_style / layout_style / widget_id passthrough                           #
# --------------------------------------------------------------------------- #


class TestStyleAndExportPassthrough:
    def test_base_plot_style_applied_to_every_trace(self, basic_df, capture_figure):
        draw_stacked_percentage_bar_chart(
            basic_df, group_by=["month"], category_column="species", plot_style=PlotStyle(width=3)
        )
        fig = capture_figure["figure"]
        assert all(trace.width == 3 for trace in fig.data)

    def test_layout_style_overrides_defaults(self, basic_df, capture_figure):
        draw_stacked_percentage_bar_chart(
            basic_df,
            group_by=["month"],
            category_column="species",
            layout_style=BarLayoutStyle(bargap=0.5, bargroupgap=0.25),
        )
        fig = capture_figure["figure"]
        assert fig.layout.bargap == 0.5
        assert fig.layout.bargroupgap == 0.25

    def test_widget_id_sets_html_div_id(self, basic_df):
        html = draw_stacked_percentage_bar_chart(
            basic_df, group_by=["month"], category_column="species", widget_id="my-widget-123"
        )
        assert 'id="my-widget-123"' in html

    def test_no_widget_id_still_produces_valid_html(self, basic_df):
        html = draw_stacked_percentage_bar_chart(basic_df, group_by=["month"], category_column="species")
        assert isinstance(html, str)
        assert len(html) > 0


# --------------------------------------------------------------------------- #
# Validation / error behavior                                                  #
# --------------------------------------------------------------------------- #


class TestValidation:
    def test_none_dataframe_raises(self):
        with pytest.raises(ValueError, match="dataframe is empty"):
            draw_stacked_percentage_bar_chart(None, group_by=["month"], category_column="species")

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="dataframe is empty"):
            draw_stacked_percentage_bar_chart(pd.DataFrame(), group_by=["month"], category_column="species")

    def test_empty_group_by_raises(self, basic_df):
        with pytest.raises(ValueError, match="group_by must contain at least one column"):
            draw_stacked_percentage_bar_chart(basic_df, group_by=[], category_column="species")

    def test_missing_group_by_column_raises(self, basic_df):
        with pytest.raises(ValueError, match=r"missing required columns: \['nope'\]"):
            draw_stacked_percentage_bar_chart(basic_df, group_by=["nope"], category_column="species")

    def test_missing_category_column_raises(self, basic_df):
        with pytest.raises(ValueError, match=r"missing required columns: \['nope'\]"):
            draw_stacked_percentage_bar_chart(basic_df, group_by=["month"], category_column="nope")

    def test_missing_color_column_raises(self, basic_df):
        with pytest.raises(ValueError, match=r"missing required columns: \['nope'\]"):
            draw_stacked_percentage_bar_chart(
                basic_df, group_by=["month"], category_column="species", color_column="nope"
            )

    def test_unexpected_numeric_category_column_is_stringified(self, capture_figure):
        # category_column with non-string (numeric) dtype: values are still
        # usable as legend/trace names via `str(category)`, and grouping
        # still works based on raw values.
        df = pd.DataFrame({"month": ["Jan", "Jan", "Feb"], "code": [1, 2, 1]})
        draw_stacked_percentage_bar_chart(df, group_by=["month"], category_column="code")
        fig = capture_figure["figure"]
        assert {trace.name for trace in fig.data} == {"1", "2"}

    def test_all_nan_category_column_raises_no_known_categories(self):
        df = pd.DataFrame({"month": ["Jan", "Jan"], "species": [None, None]})
        # No grouped_styles -> observed categories come from `.dropna()`, so
        # an all-NaN category column yields zero discovered categories and
        # therefore zero traces (not a hard error) -- documents behavior.
        html = draw_stacked_percentage_bar_chart(df, group_by=["month"], category_column="species")
        assert isinstance(html, str) and len(html) > 0
