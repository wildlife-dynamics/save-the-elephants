"""Tests for ecoscope_workflows_ext_ste.tasks.reporting._cover_context.

Covers the plain helper `get_image_dimensions_from_pixels` plus the three
`wt_registry.register()`-decorated functions (`register()` is a no-op at call
time, so these are exercised as ordinary Python functions):
    - create_context_page
    - prepare_cover_metadata
    - build_extra_fields
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from ecoscope.platform.tasks.filter._filter import TimeRange
from ecoscope_workflows_ext_ste.tasks.reporting._cover_context import (
    build_extra_fields,
    create_context_page,
    get_image_dimensions_from_pixels,
    prepare_cover_metadata,
)


class TestGetImageDimensionsFromPixels:
    def test_wide_image_scales_by_width(self, make_png):
        # 192x96 px @ 96 dpi (no dpi metadata) -> 2in x 1in native; width > height
        # so scale = max_dim / width = 1.5 / 2 = 0.75
        path = make_png(size=(192, 96))
        width, height = get_image_dimensions_from_pixels(str(path), dpi=96, max_dimension_inches=1.5)
        assert width == pytest.approx(1.5)
        assert height == pytest.approx(0.75)

    def test_tall_image_scales_by_height(self, make_png):
        # 96x192 px @ 96 dpi -> 1in x 2in native; height > width
        # so scale = max_dim / height = 1.5 / 2 = 0.75
        path = make_png(size=(96, 192))
        width, height = get_image_dimensions_from_pixels(str(path), dpi=96, max_dimension_inches=1.5)
        assert width == pytest.approx(0.75)
        assert height == pytest.approx(1.5)

    def test_square_image_uses_height_branch(self, make_png):
        # width_inches == height_inches, so `width_inches > height_inches` is False
        # and the else branch (scale by height) is used; result is square either way.
        path = make_png(size=(96, 96))
        width, height = get_image_dimensions_from_pixels(str(path), dpi=96, max_dimension_inches=1.5)
        assert width == pytest.approx(1.5)
        assert height == pytest.approx(1.5)

    def test_uses_embedded_dpi_metadata_over_default_param(self, make_png):
        # Image saved with explicit dpi=(150,150); the function should prefer
        # that over the `dpi=96` default parameter.
        path = make_png(size=(300, 150), dpi=(150, 150))
        width, height = get_image_dimensions_from_pixels(str(path), dpi=96, max_dimension_inches=1.5)
        # native: 300/150=2in x 150/150=1in -> width>height -> scale=1.5/2=0.75
        assert width == pytest.approx(1.5)
        assert height == pytest.approx(0.75)

    def test_falls_back_to_dpi_param_when_no_metadata(self, make_png):
        path = make_png(size=(480, 240))  # no dpi info saved
        width, height = get_image_dimensions_from_pixels(str(path), dpi=240, max_dimension_inches=1.5)
        # native: 480/240=2in x 240/240=1in -> same shape as above
        assert width == pytest.approx(1.5)
        assert height == pytest.approx(0.75)

    def test_non_tuple_dpi_info_is_applied_to_both_axes(self, make_png):
        # PIL always reports a 2-tuple for PNG dpi in practice, so exercise the
        # `else` branch (scalar dpi) by mocking Image.open's returned info dict.
        path = make_png(size=(200, 100))

        class _FakeImg:
            size = (200, 100)
            info = {"dpi": 100}  # scalar, not a tuple

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with patch(
            "ecoscope_workflows_ext_ste.tasks.reporting._cover_context.Image.open",
            return_value=_FakeImg(),
        ):
            width, height = get_image_dimensions_from_pixels(str(path), dpi=96, max_dimension_inches=1.5)
        # native: 200/100=2in x 100/100=1in -> width>height -> scale=1.5/2=0.75
        assert width == pytest.approx(1.5)
        assert height == pytest.approx(0.75)

    def test_max_dimension_inches_is_respected(self, make_png):
        path = make_png(size=(192, 96))
        width, height = get_image_dimensions_from_pixels(str(path), dpi=96, max_dimension_inches=3.0)
        assert width == pytest.approx(3.0)
        assert height == pytest.approx(1.5)


class TestCreateContextPage:
    def test_happy_path_default_filename(self, make_docx_template, tmp_path, read_docx_text):
        template = make_docx_template(["Prepared by: {{ prepared_by }}", "Period: {{ report_period }}"])
        output_dir = tmp_path / "out"
        context = {"prepared_by": "Tevin", "report_period": "Jan to Feb"}

        result_path = create_context_page(
            template_path=str(template),
            output_dir=str(output_dir),
            context=context,
        )

        assert result_path == str(output_dir / "cover_page.docx")
        assert os.path.exists(result_path)
        texts = read_docx_text(Path(result_path))
        assert "Prepared by: Tevin" in texts
        assert "Period: Jan to Feb" in texts

    def test_custom_filename(self, make_docx_template, tmp_path):
        template = make_docx_template(["Hello {{ name }}"])
        output_dir = tmp_path / "out"

        result_path = create_context_page(
            template_path=str(template),
            output_dir=str(output_dir),
            context={"name": "World"},
            filename="my_cover.docx",
        )

        assert result_path == str(output_dir / "my_cover.docx")
        assert os.path.exists(result_path)

    def test_creates_output_dir_if_missing(self, make_docx_template, tmp_path):
        template = make_docx_template(["static text"])
        output_dir = tmp_path / "does" / "not" / "exist" / "yet"
        assert not output_dir.exists()

        result_path = create_context_page(
            template_path=str(template),
            output_dir=str(output_dir),
            context={},
        )

        assert output_dir.exists()
        assert os.path.exists(result_path)

    def test_org_logo_embedded_when_path_exists(self, make_docx_template, make_png, tmp_path, read_docx_text):
        logo = make_png(size=(200, 100))
        template = make_docx_template(["Logo: {{ org_logo }}"])
        output_dir = tmp_path / "out"

        context = {"org_logo_path": str(logo)}
        result_path = create_context_page(
            template_path=str(template),
            output_dir=str(output_dir),
            context=context,
        )

        assert os.path.exists(result_path)
        # docxtpl replaces the placeholder text with an image run, so the
        # literal "{{ org_logo }}" text is gone but the surrounding text remains.
        texts = read_docx_text(Path(result_path))
        assert any(t.startswith("Logo:") for t in texts)
        assert not any("{{ org_logo }}" in t for t in texts)

    def test_missing_org_logo_path_renders_without_error(self, make_docx_template, tmp_path):
        template = make_docx_template(["Logo: {{ org_logo }}"])
        output_dir = tmp_path / "out"

        # org_logo_path points at a file that does not exist -> the
        # `os.path.exists(...)` guard is False, so no InlineImage is built,
        # and the template renders with an empty/undefined value instead of
        # raising.
        context = {"org_logo_path": str(tmp_path / "nope.png")}
        result_path = create_context_page(
            template_path=str(template),
            output_dir=str(output_dir),
            context=context,
        )
        assert os.path.exists(result_path)

    def test_file_scheme_paths_are_normalized(self, make_docx_template, tmp_path):
        template = make_docx_template(["hi {{ x }}"])
        output_dir = tmp_path / "out"

        result_path = create_context_page(
            template_path="file://" + str(template),
            output_dir="file://" + str(output_dir),
            context={"x": "1"},
        )
        assert os.path.exists(result_path)
        assert not result_path.startswith("file://")


class TestPrepareCoverMetadata:
    def test_core_fields_present_with_no_logo(self, make_time_range):
        report_period = make_time_range()
        result = prepare_cover_metadata(
            org_logo_path=None,
            report_period=report_period,
            prepared_by="Jane Doe",
        )
        assert result["org_logo_path"] is None
        assert result["prepared_by"] == "Jane Doe"
        # TimeRange.time_format defaults to "%d %b %Y %H:%M:%S" (DEFAULT_TIME_FORMAT),
        # not the "%Y-%m-%d" fallback in prepare_cover_metadata's getattr call --
        # that fallback only matters for objects without a time_format attribute.
        assert result["report_period"] == "01 Jan 2024 00:00:00 to 01 Feb 2024 00:00:00"
        # time_generated should be a valid, recent timestamp in the default format
        parsed = datetime.strptime(result["time_generated"], "%Y-%m-%d %H:%M:%S")
        assert abs((datetime.now() - parsed).total_seconds()) < 60

    def test_report_period_uses_time_range_time_format(self, make_time_range):
        report_period = make_time_range(time_format="%Y/%m/%d")
        result = prepare_cover_metadata(
            org_logo_path=None,
            report_period=report_period,
            prepared_by="Jane Doe",
        )
        assert result["report_period"] == "2024/01/01 to 2024/02/01"

    def test_report_period_fallback_format_for_objects_without_time_format(self):
        # prepare_cover_metadata's type hint says TimeRange, but the getattr(...,
        # "time_format", "%Y-%m-%d") fallback only kicks in for objects that
        # lack the attribute entirely -- exercise that branch directly with a
        # minimal duck-typed stand-in.
        from types import SimpleNamespace

        fake_period = SimpleNamespace(since=datetime(2024, 3, 5), until=datetime(2024, 3, 6))
        result = prepare_cover_metadata(
            org_logo_path=None,
            report_period=fake_period,
            prepared_by="X",
        )
        assert result["report_period"] == "2024-03-05 to 2024-03-06"

    def test_org_logo_path_normalized(self, make_time_range, tmp_path):
        report_period = make_time_range()
        logo = tmp_path / "logo.png"
        result = prepare_cover_metadata(
            org_logo_path="file://" + str(logo),
            report_period=report_period,
            prepared_by="X",
        )
        assert result["org_logo_path"] == str(logo)

    def test_empty_org_logo_path_raises(self, make_time_range):
        report_period = make_time_range()
        with pytest.raises(ValueError, match="org_logo_path is empty"):
            prepare_cover_metadata(
                org_logo_path="",
                report_period=report_period,
                prepared_by="X",
            )

    def test_extra_fields_are_merged_and_stringified(self, make_time_range):
        report_period = make_time_range()
        result = prepare_cover_metadata(
            org_logo_path=None,
            report_period=report_period,
            prepared_by="X",
            extra_fields={"subject_count": 5, "title": "Q1 Report", "note": None},
        )
        assert result["subject_count"] == "5"
        assert result["title"] == "Q1 Report"
        assert result["note"] is None

    def test_extra_fields_can_override_core_fields(self, make_time_range):
        report_period = make_time_range()
        result = prepare_cover_metadata(
            org_logo_path=None,
            report_period=report_period,
            prepared_by="Original",
            extra_fields={"prepared_by": "Overridden"},
        )
        assert result["prepared_by"] == "Overridden"

    def test_custom_time_generated_format(self, make_time_range):
        report_period = make_time_range()
        result = prepare_cover_metadata(
            org_logo_path=None,
            report_period=report_period,
            prepared_by="X",
            time_generated_format="%Y",
        )
        assert result["time_generated"] == str(datetime.now().year)

    def test_report_period_requires_matching_tz_awareness(self, utc_timezone):
        # TimeRange's model_validator rejects mixed naive/aware since/until;
        # this is upstream behavior that prepare_cover_metadata relies on
        # (report_period is always a valid, tz-consistent TimeRange by the
        # time it reaches this function).
        naive = datetime(2024, 1, 1)
        aware = datetime(2024, 2, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="timezone naive, or both aware"):
            TimeRange(since=naive, until=aware, timezone=utc_timezone)


class TestBuildExtraFields:
    def test_stringifies_values(self):
        result = build_extra_fields({"count": 3, "ratio": 1.5, "flag": True})
        assert result == {"count": "3", "ratio": "1.5", "flag": "True"}

    def test_preserves_none(self):
        result = build_extra_fields({"missing": None, "present": "value"})
        assert result["missing"] is None
        assert result["present"] == "value"

    def test_empty_mapping(self):
        assert build_extra_fields({}) == {}

    def test_already_string_values_pass_through(self):
        result = build_extra_fields({"title": "Report"})
        assert result["title"] == "Report"
