"""Tests for ecoscope_workflows_ext_ste.tasks._log."""

import pytest

from ecoscope_workflows_ext_ste.tasks._log import STAGES, count, log, timed


def test_log_prefixes_the_stage_name(capsys):
    log("seasons", "14 requests")

    assert capsys.readouterr().out == "[STE Seasons] 14 requests\n"


def test_log_with_an_unknown_stage_uses_it_as_given(capsys):
    log("Something else", "hello")

    assert capsys.readouterr().out == "[STE Something else] hello\n"


def test_every_stage_has_a_label():
    assert all(isinstance(label, str) and label for label in STAGES.values())


@pytest.mark.parametrize(
    "n, noun, plural, expected",
    [
        (1, "segment", None, "1 segment"),
        (0, "segment", None, "0 segments"),
        (1556, "segment", None, "1,556 segments"),
        (1, "category", "categories", "1 category"),
        (3, "category", "categories", "3 categories"),
    ],
)
def test_count(n, noun, plural, expected):
    assert count(n, noun, plural) == expected


def test_timed_prints_start_and_duration(capsys):
    with timed("ranges", "home range"):
        pass

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[STE Home range] home range..."
    assert lines[1].startswith("[STE Home range] home range: done in ")
    assert lines[1].endswith("s")


def test_timed_reports_a_failure_and_reraises(capsys):
    with pytest.raises(ValueError, match="boom"):
        with timed("seasons", "NDVI"):
            raise ValueError("boom")

    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("[STE Seasons] NDVI: failed after ")
    assert lines[1].endswith("(ValueError)")
