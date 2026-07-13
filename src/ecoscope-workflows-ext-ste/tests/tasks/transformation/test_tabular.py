"""Tests for ecoscope_workflows_ext_ste.tasks.transformation._tabular.

All seven functions here are registered via `wt_registry.register()`, which
is a no-op at call time, so they behave as plain Python functions in these
tests.

Note on in-place mutation: `subset_columns` and `add_new_column` both return
a `.copy()` of the input, but `add_mapped_column_value` and
`convert_columns_to_string` mutate the passed-in DataFrame directly (no
`.copy()`) before returning it. That inconsistency is exercised explicitly
below (see `test_mutates_input_df_in_place` in each class) and reported as a
suspected footgun rather than "fixed" here.
"""

import numpy as np
import pandas as pd
import pytest

from ecoscope_workflows_ext_ste.tasks.transformation._tabular import (
    add_mapped_column_value,
    add_new_column,
    column_first_unique_value,
    convert_columns_to_string,
    round_off_values,
    safe_string,
    subset_columns,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["Nairobi", "Mombasa", "Kisumu"],
        }
    )


class TestSubsetColumns:
    def test_no_filter_returns_all_columns_as_copy(self, sample_df):
        result = subset_columns(sample_df)

        assert list(result.columns) == list(sample_df.columns)
        assert result.equals(sample_df)
        assert result is not sample_df

    def test_columns_allowlist_selects_and_orders(self, sample_df):
        result = subset_columns(sample_df, columns=["city", "name"])
        assert list(result.columns) == ["city", "name"]
        assert len(result) == len(sample_df)

    def test_columns_returns_copy_not_view(self, sample_df):
        result = subset_columns(sample_df, columns=["name"])
        result.loc[0, "name"] = "Mutated"
        assert sample_df.loc[0, "name"] == "Alice"

    def test_missing_columns_warn_and_are_skipped(self, sample_df):
        with pytest.warns(UserWarning, match=r"Columns not found.*nonexistent"):
            result = subset_columns(sample_df, columns=["name", "nonexistent"])
        assert list(result.columns) == ["name"]

    def test_missing_columns_strict_raises_key_error(self, sample_df):
        with pytest.raises(KeyError):
            subset_columns(sample_df, columns=["name", "nonexistent"], strict=True)

    def test_exclude_denylist_preserves_remaining_order(self, sample_df):
        result = subset_columns(sample_df, exclude=["age"])
        assert list(result.columns) == ["name", "city"]

    def test_exclude_missing_warns_but_keeps_all_columns(self, sample_df):
        with pytest.warns(UserWarning, match=r"Columns not found, skipping exclusion"):
            result = subset_columns(sample_df, exclude=["nonexistent"])
        assert list(result.columns) == list(sample_df.columns)

    def test_exclude_missing_strict_raises_key_error(self, sample_df):
        with pytest.raises(KeyError):
            subset_columns(sample_df, exclude=["nonexistent"], strict=True)

    def test_both_columns_and_exclude_raises_value_error(self, sample_df):
        with pytest.raises(ValueError, match="Pass either"):
            subset_columns(sample_df, columns=["name"], exclude=["age"])

    def test_empty_columns_list_returns_no_columns(self, sample_df):
        result = subset_columns(sample_df, columns=[])
        assert list(result.columns) == []
        assert len(result) == len(sample_df)

    def test_empty_exclude_list_returns_all_columns(self, sample_df):
        result = subset_columns(sample_df, exclude=[])
        assert list(result.columns) == list(sample_df.columns)


class TestAddMappedColumnValue:
    def test_basic_mapping_with_default_new_column_name(self, sample_df):
        result = add_mapped_column_value(sample_df, "city", {"Nairobi": "NBO", "Mombasa": "MBA"})

        assert "city_mapped" in result.columns
        assert result.loc[0, "city_mapped"] == "NBO"
        assert result.loc[1, "city_mapped"] == "MBA"

    def test_unmapped_values_default_to_none(self, sample_df):
        result = add_mapped_column_value(sample_df.copy(), "city", {"Nairobi": "NBO"})
        assert pd.isna(result.loc[2, "city_mapped"])  # "Kisumu" is unmapped

    def test_custom_new_column_name(self, sample_df):
        result = add_mapped_column_value(sample_df.copy(), "city", {"Nairobi": "NBO"}, new_column="city_code")
        assert "city_code" in result.columns
        assert "city_mapped" not in result.columns

    def test_default_value_used_for_unmapped(self, sample_df):
        result = add_mapped_column_value(sample_df.copy(), "city", {"Nairobi": "NBO"}, default="UNKNOWN")
        assert result.loc[2, "city_mapped"] == "UNKNOWN"

    def test_keep_unmapped_passes_through_original_value(self, sample_df):
        result = add_mapped_column_value(sample_df.copy(), "city", {"Nairobi": "NBO"}, keep_unmapped=True)
        assert result.loc[2, "city_mapped"] == "Kisumu"

    def test_keep_unmapped_true_ignores_default(self, sample_df):
        # docstring states `default` is ignored when `keep_unmapped=True`
        result = add_mapped_column_value(
            sample_df.copy(),
            "city",
            {"Nairobi": "NBO"},
            default="UNKNOWN",
            keep_unmapped=True,
        )
        assert result.loc[2, "city_mapped"] == "Kisumu"

    def test_missing_column_raises_key_error(self, sample_df):
        with pytest.raises(KeyError):
            add_mapped_column_value(sample_df.copy(), "nonexistent", {"a": "b"})

    def test_mutates_input_df_in_place(self, sample_df):
        df = sample_df.copy()
        out = add_mapped_column_value(df, "city", {"Nairobi": "NBO"})
        # unlike `subset_columns`/`add_new_column`, this does not copy first
        assert out is df
        assert "city_mapped" in df.columns

    def test_empty_mapping_all_values_unmapped(self, sample_df):
        result = add_mapped_column_value(sample_df.copy(), "city", {})
        assert result["city_mapped"].isna().all()


class TestAddNewColumn:
    def test_adds_int_column(self, sample_df):
        result = add_new_column(sample_df, "new_col", 42)
        assert "new_col" in result.columns
        assert (result["new_col"] == 42).all()
        assert len(result.columns) == len(sample_df.columns) + 1

    def test_adds_float_column(self, sample_df):
        result = add_new_column(sample_df, "score", 98.5)
        assert (result["score"] == 98.5).all()

    def test_adds_string_column(self, sample_df):
        result = add_new_column(sample_df, "status", "active")
        assert (result["status"] == "active").all()

    def test_existing_column_is_left_unchanged(self, sample_df):
        original_values = sample_df["age"].copy()
        result = add_new_column(sample_df, "age", 999)
        assert (result["age"] == original_values).all()
        assert not (result["age"] == 999).any()

    def test_returns_copy_original_df_unchanged(self, sample_df):
        original_cols = sample_df.columns.tolist()
        _ = add_new_column(sample_df, "new_col", 100)
        assert sample_df.columns.tolist() == original_cols
        assert "new_col" not in sample_df.columns


class TestColumnFirstUniqueValue:
    def test_first_unique_value_basic(self, sample_df):
        result = column_first_unique_value(sample_df, "name")
        assert result == "Alice"
        assert isinstance(result, str)

    def test_numeric_column_converted_to_string(self):
        df = pd.DataFrame({"year": [2024, 2023, 2022]})
        result = column_first_unique_value(df, "year")
        assert result == "2024"
        assert isinstance(result, str)

    def test_does_not_apply_sentence_case_despite_docs(self):
        # Suspected doc/implementation mismatch: the return annotation's
        # Field description promises "(sentence case)" output, but the
        # implementation is a bare `str(unique_values[0])` with no case
        # transformation at all.
        df = pd.DataFrame({"status": ["ACTIVE", "inactive", "pending"]})
        result = column_first_unique_value(df, "status")
        assert result == "ACTIVE"  # NOT "Active"

        df2 = pd.DataFrame({"county": ["turkana", "marsabit"]})
        result2 = column_first_unique_value(df2, "county")
        assert result2 == "turkana"  # NOT "Turkana"

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="df is empty"):
            column_first_unique_value(pd.DataFrame(), "any_column")

    def test_none_dataframe_raises(self):
        with pytest.raises(ValueError, match="df is empty"):
            column_first_unique_value(None, "any_column")

    def test_nonexistent_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            column_first_unique_value(sample_df, "nonexistent")

    def test_column_with_empty_string_first_value(self):
        df = pd.DataFrame({"status": ["", "active", "pending"]})
        result = column_first_unique_value(df, "status")
        assert result == ""

    def test_all_null_column_does_not_raise(self):
        # unique() on an all-None column still returns one element (None), so
        # the "no values found" branch is never hit; str(None) == "None".
        df = pd.DataFrame({"col": [None, None, None]})
        result = column_first_unique_value(df, "col")
        assert result == "None"


class TestConvertColumnsToString:
    def test_single_column_as_bare_string(self, sample_df):
        result = convert_columns_to_string(sample_df.copy(), "age")
        assert result["age"].dtype == object
        assert result.loc[0, "age"] == "25"

    def test_multiple_columns_as_list(self, sample_df):
        result = convert_columns_to_string(sample_df.copy(), ["age", "name"])
        assert result["age"].dtype == object
        assert result["name"].dtype == object

    def test_missing_column_is_skipped_not_raised(self, sample_df, capsys):
        result = convert_columns_to_string(sample_df.copy(), ["nonexistent"])
        captured = capsys.readouterr()
        assert "not found" in captured.out
        assert list(result.columns) == list(sample_df.columns)

    def test_mutates_input_df_in_place(self, sample_df):
        df = sample_df.copy()
        out = convert_columns_to_string(df, "age")
        assert out is df
        assert df["age"].dtype == object

    def test_mixed_existing_and_missing_columns(self, sample_df, capsys):
        result = convert_columns_to_string(sample_df.copy(), ["age", "missing_col"])
        assert result["age"].dtype == object
        captured = capsys.readouterr()
        assert "missing_col" in captured.out


class TestSafeString:
    def test_replaces_spaces_with_underscores(self):
        assert safe_string("Hello World") == "hello_world"

    def test_removes_special_characters(self):
        assert safe_string("Hello, World!") == "hello_world"

    def test_lowercases(self):
        assert safe_string("UPPER CASE") == "upper_case"

    def test_collapses_multiple_spaces(self):
        assert safe_string("  multiple   spaces  ") == "multiple_spaces"

    def test_strips_leading_and_trailing_underscores(self):
        assert safe_string("___lead_trail___") == "lead_trail"

    def test_empty_string_returns_empty(self):
        assert safe_string("") == ""

    def test_all_special_characters_returns_empty(self):
        assert safe_string("###@@@") == ""

    def test_preserves_hyphens_digits_and_underscores(self):
        assert safe_string("a-b_c 123") == "a-b_c_123"

    def test_preserves_unicode_word_characters(self):
        assert safe_string("Café Résumé") == "café_résumé"


class TestRoundOffValues:
    def test_round_to_zero_decimals(self):
        result = round_off_values(3.7, 0)
        assert result == 4.0
        assert isinstance(result, float)

    def test_round_to_one_decimal(self):
        assert round_off_values(3.14159, 1) == 3.1

    def test_round_to_two_decimals(self):
        assert round_off_values(3.14159, 2) == 3.14

    def test_round_to_five_decimals(self):
        assert round_off_values(3.14159265359, 5) == 3.14159

    def test_round_negative_number(self):
        assert round_off_values(-3.14159, 2) == -3.14

    def test_round_zero(self):
        assert round_off_values(0.0, 2) == 0.0

    def test_round_already_rounded(self):
        assert round_off_values(5.5, 1) == 5.5

    def test_round_integer_as_float(self):
        assert round_off_values(42.0, 2) == 42.0

    def test_round_half_to_even_banker_rounding(self):
        assert round_off_values(2.5, 0) == 2.0
        assert round_off_values(3.5, 0) == 4.0

    def test_round_very_small_number(self):
        assert round_off_values(0.000123456, 6) == 0.000123

    def test_round_very_large_number(self):
        assert round_off_values(123456789.987654321, 2) == 123456789.99

    def test_round_with_negative_decimal_places(self):
        assert round_off_values(12345.67, -1) == 12350.0
        assert round_off_values(12345.67, -2) == 12300.0

    def test_round_nan_returns_nan(self):
        result = round_off_values(float("nan"), 2)
        assert np.isnan(result)
