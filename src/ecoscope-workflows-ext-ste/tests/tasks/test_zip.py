from ecoscope_workflows_ext_ste.tasks import zip_grouped_by_key


def test_zip_grouped_by_key_basic():
    left = [("A", 1), ("B", 2), ("C", 3)]
    right = [("B", 20), ("C", 30), ("A", 10)]
    out = zip_grouped_by_key(left=left, right=right)
    assert out == [
        ("A", (1, 10)),
        ("B", (2, 20)),
        ("C", (3, 30)),
    ]


def test_zip_grouped_by_key_missing_keys_are_dropped():
    left = [("A", 1), ("B", 2), ("C", 3)]
    right = [("B", 20)]  # A and C missing
    out = zip_grouped_by_key(left=left, right=right)
    assert out == [("B", (2, 20))]


def test_zip_grouped_by_key_preserves_left_order():
    left = [("X", 100), ("Y", 200), ("Z", 300)]
    right = [("Z", 3), ("Y", 2), ("X", 1)]
    out = zip_grouped_by_key(left=left, right=right)
    assert [k for k, _ in out] == ["X", "Y", "Z"]


def test_zip_grouped_by_key_ignores_extras_in_right():
    left = [("dog", 1)]
    right = [("dog", 10), ("cat", 20), ("fish", 30)]
    out = zip_grouped_by_key(left=left, right=right)
    assert out == [("dog", (1, 10))]


def test_zip_grouped_by_key_empty_inputs():
    assert zip_grouped_by_key(left=[], right=[]) == []
    assert zip_grouped_by_key(left=[("A", 1)], right=[]) == []
