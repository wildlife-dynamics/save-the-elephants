"""Tests for the keyed-iterable tasks in ecoscope_workflows_ext_ste.tasks.skip._groupby."""

from wt_task.skip import SKIP_SENTINEL

from ecoscope_workflows_ext_ste.tasks.skip import groupbykey_complete, groupbykey_keep_skips

A = (("subject_name", "=", "A"),)
B = (("subject_name", "=", "B"),)
C = (("subject_name", "=", "C"),)


def test_combines_values_of_complete_groups_in_order():
    result = groupbykey_complete(iterables=[[(A, "a.html"), (B, "b.html")], [(A, 1), (B, 2)]])

    assert result == [(A, ["a.html", 1]), (B, ["b.html", 2])]


def test_group_with_a_skipped_value_is_left_out_and_others_kept():
    """One subject without a map must not remove the map for the others."""
    result = groupbykey_complete(
        iterables=[
            [(A, "a.html"), (B, "b.html"), (C, "c.html")],
            [(A, 1), (B, SKIP_SENTINEL), (C, 3)],
        ]
    )

    assert result == [(A, ["a.html", 1]), (C, ["c.html", 3])]


def test_group_missing_from_one_iterable_is_left_out():
    result = groupbykey_complete(iterables=[[(A, "a.html"), (B, "b.html")], [(A, 1)]])

    assert result == [(A, ["a.html", 1])]


def test_none_values_and_null_slots_count_as_missing():
    result = groupbykey_complete(
        iterables=[
            [(A, "a.html"), (B, "b.html"), None, (None, "x")],
            [(A, 1), (B, None)],
        ]
    )

    assert result == [(A, ["a.html", 1])]


def test_three_iterables_need_a_value_in_each():
    result = groupbykey_complete(
        iterables=[
            [(A, "df-a"), (B, "df-b")],
            [(A, 10), (B, 20)],
            [(A, SKIP_SENTINEL), (B, 200)],
        ]
    )

    assert result == [(B, ["df-b", 20, 200])]


def test_no_complete_group_returns_skip_sentinel():
    result = groupbykey_complete(iterables=[[(A, "a.html")], [(A, SKIP_SENTINEL)]])

    assert result is SKIP_SENTINEL


def test_falsy_values_are_real_values():
    result = groupbykey_complete(iterables=[[(A, 0), (B, "")], [(A, []), (B, False)]])

    assert result == [(A, [0, []]), (B, ["", False])]


def test_keep_skips_keeps_every_group_with_skip_markers():
    result = groupbykey_keep_skips(
        iterables=[
            [(A, "df-a"), (B, "df-b")],
            [(A, 10), (B, SKIP_SENTINEL)],
            [(A, 100)],
        ]
    )

    assert result == [(A, ["df-a", 10, 100]), (B, ["df-b", SKIP_SENTINEL, SKIP_SENTINEL])]


def test_keep_skips_with_no_groups_returns_skip_sentinel():
    assert groupbykey_keep_skips(iterables=[[], [None]]) is SKIP_SENTINEL
