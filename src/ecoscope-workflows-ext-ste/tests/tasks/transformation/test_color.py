"""Tests for ecoscope_workflows_ext_ste.tasks.transformation._color.

Both `add_status_color_columns` and `add_rgba_from_hex` are registered via
`wt_registry.register()`, which is a no-op at call time, so they behave as
plain Python functions here -- no pydantic validation/coercion happens on
the way in.

`hex_to_rgba` (from `ecoscope.base.utils`) is the underlying hex->RGBA
converter used by both functions. Its documented contract (confirmed by
reading its source):
    - raises ValueError("Input cannot be empty") for falsy input
    - strips a leading "#" if present
    - requires the remaining string to be 6 or 8 hex characters, else
      raises ValueError("Invalid hex length, must be 6 or 8")
    - a 6-char string is padded with "FF" (full opacity) before conversion
    - non-hex characters raise ValueError(f"Invalid hex string, {input}")
Known conversions used below as ground truth (verified directly against
`hex_to_rgba` before being hardcoded as expectations):
    "#000000" -> (0, 0, 0, 255)
    "#ffffff" -> (255, 255, 255, 255)
    "ff0000"  -> (255, 0, 0, 255)   (no leading "#" required)
    "#ff000080" -> (255, 0, 0, 128)
"""

import pandas as pd
import pytest

from ecoscope_workflows_ext_ste.tasks.transformation._color import (
    add_rgba_from_hex,
    add_status_color_columns,
)


class TestAddRgbaFromHexHappyPath:
    def test_adds_new_column_with_expected_tuples(self):
        df = pd.DataFrame({"hex": ["#000000", "#ffffff", "ff0000"]})
        result = add_rgba_from_hex(df, column="hex", new_column="rgba")
        assert list(result["rgba"]) == [
            (0, 0, 0, 255),
            (255, 255, 255, 255),
            (255, 0, 0, 255),
        ]

    def test_preserves_alpha_channel_when_provided(self):
        df = pd.DataFrame({"hex": ["#ff000080"]})
        result = add_rgba_from_hex(df, column="hex", new_column="rgba")
        assert result["rgba"].iloc[0] == (255, 0, 0, 128)

    def test_does_not_mutate_input_df(self):
        df = pd.DataFrame({"hex": ["#000000"]})
        original_columns = list(df.columns)
        add_rgba_from_hex(df, column="hex", new_column="rgba")
        assert list(df.columns) == original_columns

    def test_original_column_untouched(self):
        df = pd.DataFrame({"hex": ["#000000", "#ffffff"]})
        result = add_rgba_from_hex(df, column="hex", new_column="rgba")
        assert list(result["hex"]) == ["#000000", "#ffffff"]

    def test_returns_dataframe_type(self):
        df = pd.DataFrame({"hex": ["#000000"]})
        result = add_rgba_from_hex(df, column="hex", new_column="rgba")
        assert isinstance(result, pd.DataFrame)

    def test_row_by_row_conversion_not_unique_based(self):
        # Every row is converted independently (via .apply), unlike
        # add_status_color_columns which builds a lookup of unique values.
        df = pd.DataFrame({"hex": ["#000000", "#000000", "#ffffff"]})
        result = add_rgba_from_hex(df, column="hex", new_column="rgba")
        assert list(result["rgba"]) == [
            (0, 0, 0, 255),
            (0, 0, 0, 255),
            (255, 255, 255, 255),
        ]


class TestAddRgbaFromHexEdgeCases:
    def test_missing_column_raises_value_error(self):
        df = pd.DataFrame({"other": [1, 2]})
        with pytest.raises(ValueError, match="Column 'hex' not found"):
            add_rgba_from_hex(df, column="hex", new_column="rgba")

    def test_missing_column_error_lists_available_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError) as excinfo:
            add_rgba_from_hex(df, column="hex", new_column="rgba")
        message = str(excinfo.value)
        assert "'a'" in message
        assert "'b'" in message

    def test_empty_dataframe_with_column_present(self):
        df = pd.DataFrame({"hex": pd.Series([], dtype="object")})
        result = add_rgba_from_hex(df, column="hex", new_column="rgba")
        assert len(result) == 0
        assert "rgba" in result.columns

    def test_single_row(self):
        df = pd.DataFrame({"hex": ["#123456"]})
        result = add_rgba_from_hex(df, column="hex", new_column="rgba")
        assert len(result) == 1
        assert result["rgba"].iloc[0] == (0x12, 0x34, 0x56, 255)

    def test_out_of_range_hex_length_raises(self):
        df = pd.DataFrame({"hex": ["#fff"]})
        with pytest.raises(ValueError, match="Invalid hex length"):
            add_rgba_from_hex(df, column="hex", new_column="rgba")

    def test_non_hex_characters_raise(self):
        df = pd.DataFrame({"hex": ["#gggggg"]})
        with pytest.raises(ValueError, match="Invalid hex string"):
            add_rgba_from_hex(df, column="hex", new_column="rgba")

    def test_empty_string_value_raises(self):
        df = pd.DataFrame({"hex": [""]})
        with pytest.raises(ValueError, match="Input cannot be empty"):
            add_rgba_from_hex(df, column="hex", new_column="rgba")

    def test_new_column_overwrites_existing_column_of_same_name(self):
        df = pd.DataFrame({"hex": ["#000000"], "rgba": ["placeholder"]})
        result = add_rgba_from_hex(df, column="hex", new_column="rgba")
        assert result["rgba"].iloc[0] == (0, 0, 0, 255)


class TestAddStatusColorColumnsHappyPath:
    def _make_df(self):
        return pd.DataFrame(
            {
                "hex_color": ["#ff0000", "#00ff00", "#0000ff"],
                "duration_status": ["Current tracks", "Current tracks", "Past tracks"],
            }
        )

    def test_default_column_names_are_created(self):
        df = self._make_df()
        result = add_status_color_columns(df, hex_column="hex_color", previous_color_hex="#808080")
        assert "duration_status_hex_colors" in result.columns
        assert "duration_status_colors" in result.columns

    def test_current_rows_use_hex_column_when_use_hex_column_for_current(self):
        df = self._make_df()
        result = add_status_color_columns(df, hex_column="hex_color", previous_color_hex="#808080")
        # rows 0, 1 are "Current tracks" -> keep their own hex_color
        assert result["duration_status_hex_colors"].iloc[0] == "#ff0000"
        assert result["duration_status_hex_colors"].iloc[1] == "#00ff00"
        # row 2 is not current -> previous_color_hex
        assert result["duration_status_hex_colors"].iloc[2] == "#808080"

    def test_rgba_lookup_matches_hex_colors(self):
        df = self._make_df()
        result = add_status_color_columns(df, hex_column="hex_color", previous_color_hex="#808080")
        assert result["duration_status_colors"].iloc[0] == (255, 0, 0, 255)
        assert result["duration_status_colors"].iloc[1] == (0, 255, 0, 255)
        assert result["duration_status_colors"].iloc[2] == (0x80, 0x80, 0x80, 255)

    def test_use_hex_column_for_current_false_with_default_hex(self):
        df = self._make_df()
        result = add_status_color_columns(
            df,
            hex_column="hex_color",
            previous_color_hex="#808080",
            use_hex_column_for_current=False,
            default_current_hex="#00008b",
        )
        # both "current" rows get the uniform default color, not their own hex
        assert result["duration_status_hex_colors"].iloc[0] == "#00008b"
        assert result["duration_status_hex_colors"].iloc[1] == "#00008b"
        assert result["duration_status_hex_colors"].iloc[2] == "#808080"

    def test_custom_status_column_and_current_status(self):
        df = pd.DataFrame(
            {
                "hex_color": ["#ff0000", "#0000ff"],
                "state": ["active", "inactive"],
            }
        )
        result = add_status_color_columns(
            df,
            hex_column="hex_color",
            previous_color_hex="#808080",
            current_status="active",
            status_column="state",
        )
        assert "state_hex_colors" in result.columns
        assert "state_colors" in result.columns
        assert result["state_hex_colors"].iloc[0] == "#ff0000"
        assert result["state_hex_colors"].iloc[1] == "#808080"

    def test_does_not_mutate_input(self):
        df = self._make_df()
        original_columns = list(df.columns)
        add_status_color_columns(df, hex_column="hex_color", previous_color_hex="#808080")
        assert list(df.columns) == original_columns


class TestAddStatusColorColumnsFallbackBehavior:
    """Covers the fallback branch documented in-source as a design smell:

        # Consider raising here instead of silently reusing previous_color_hex.
        current_hex = previous_color_hex

    When `use_hex_column_for_current=False` and `default_current_hex=None`,
    "current" rows silently get colored with `previous_color_hex` -- i.e.
    current and previous rows become visually indistinguishable. This is
    not necessarily a bug (the source comment flags it as an intentional
    but questionable choice) but is worth pinning down with a test so a
    future change to this behavior is a deliberate decision.
    """

    def test_current_rows_fall_back_to_previous_color_hex(self):
        df = pd.DataFrame(
            {
                "hex_color": ["#ff0000", "#00ff00"],
                "duration_status": ["Current tracks", "Past tracks"],
            }
        )
        result = add_status_color_columns(
            df,
            hex_column="hex_color",
            previous_color_hex="#808080",
            use_hex_column_for_current=False,
            default_current_hex=None,
        )
        # both rows end up the same color -- current is indistinguishable
        # from previous in this fallback configuration.
        assert result["duration_status_hex_colors"].iloc[0] == "#808080"
        assert result["duration_status_hex_colors"].iloc[1] == "#808080"
        assert result["duration_status_hex_colors"].iloc[0] == result["duration_status_hex_colors"].iloc[1]


class TestAddStatusColorColumnsEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame(
            {"hex_color": pd.Series([], dtype="object"), "duration_status": pd.Series([], dtype="object")}
        )
        result = add_status_color_columns(df, hex_column="hex_color", previous_color_hex="#808080")
        assert len(result) == 0
        assert "duration_status_hex_colors" in result.columns
        assert "duration_status_colors" in result.columns

    def test_single_row_current(self):
        df = pd.DataFrame({"hex_color": ["#123456"], "duration_status": ["Current tracks"]})
        result = add_status_color_columns(df, hex_column="hex_color", previous_color_hex="#808080")
        assert result["duration_status_hex_colors"].iloc[0] == "#123456"
        assert result["duration_status_colors"].iloc[0] == (0x12, 0x34, 0x56, 255)

    def test_single_row_not_current(self):
        df = pd.DataFrame({"hex_color": ["#123456"], "duration_status": ["Past tracks"]})
        result = add_status_color_columns(df, hex_column="hex_color", previous_color_hex="#808080")
        assert result["duration_status_hex_colors"].iloc[0] == "#808080"

    def test_all_rows_current(self):
        df = pd.DataFrame(
            {
                "hex_color": ["#ff0000", "#00ff00"],
                "duration_status": ["Current tracks", "Current tracks"],
            }
        )
        result = add_status_color_columns(df, hex_column="hex_color", previous_color_hex="#808080")
        assert list(result["duration_status_hex_colors"]) == ["#ff0000", "#00ff00"]

    def test_no_rows_match_current_status(self):
        df = pd.DataFrame(
            {
                "hex_color": ["#ff0000", "#00ff00"],
                "duration_status": ["Past tracks", "Past tracks"],
            }
        )
        result = add_status_color_columns(df, hex_column="hex_color", previous_color_hex="#808080")
        assert all(result["duration_status_hex_colors"] == "#808080")

    def test_invalid_hex_value_raises(self):
        df = pd.DataFrame({"hex_color": ["#zzzzzz"], "duration_status": ["Current tracks"]})
        with pytest.raises(ValueError, match="Invalid hex string"):
            add_status_color_columns(df, hex_column="hex_color", previous_color_hex="#808080")

    def test_invalid_previous_color_hex_raises(self):
        df = pd.DataFrame({"hex_color": ["#ffffff"], "duration_status": ["Past tracks"]})
        with pytest.raises(ValueError, match="Invalid hex length"):
            add_status_color_columns(df, hex_column="hex_color", previous_color_hex="#fff")

    def test_repeated_hex_values_are_only_converted_once_via_lookup(self):
        # rgba_lookup is built from unique hex values, so duplicate colors
        # across many rows should still map correctly.
        df = pd.DataFrame(
            {
                "hex_color": ["#ff0000"] * 5,
                "duration_status": ["Current tracks"] * 5,
            }
        )
        result = add_status_color_columns(df, hex_column="hex_color", previous_color_hex="#808080")
        assert all(result["duration_status_colors"] == (255, 0, 0, 255))
