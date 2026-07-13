"""Tests for ecoscope_workflows_ext_ste.tasks.transformation._concat.

`concatenate_dataframes` is registered via `wt_registry.register()`, which
is a no-op at call time, so it behaves as a plain Python function here.

Behavior confirmed by reading the source:
    - Any `SkipSentinel` entries in `list_df` are filtered out first.
    - If nothing is left after filtering, an *empty* `pd.DataFrame()` is
      returned (not None, not an error).
    - Otherwise `pd.concat(valid_dfs, **kwargs)` is called, with
      `ignore_index=True` as the (non-pandas-default) default.
    - `copy` is only forwarded to `pd.concat` when explicitly not None, to
      avoid the pandas 3.x deprecation warning for the (no-op) `copy` kwarg.

Pandas-version-specific behavior verified empirically against this repo's
pinned pandas (2.3.3) before being encoded as an expectation:
    - Passing `keys=[...]` together with the default `ignore_index=True`
      does *not* raise and does *not* build a MultiIndex -- `keys` is
      silently a no-op whenever `ignore_index=True`. Since
      `concatenate_dataframes` defaults `ignore_index` to `True` (unlike
      `pandas.concat`, which defaults to `False`), callers who pass `keys`
      without also passing `ignore_index=False` will silently get no
      hierarchical index. This isn't exercised as a "bug" (pandas' own
      documented interaction), just pinned down as a caller-facing gotcha.
"""

import geopandas as gpd
import pandas as pd
import pytest
from wt_task.skip import SkipSentinel

from ecoscope_workflows_ext_ste.tasks.transformation._concat import (
    concatenate_dataframes,
)


def _df(**cols):
    return pd.DataFrame(cols)


class TestHappyPath:
    def test_basic_row_concat_default_axis(self):
        df1 = _df(a=[1, 2], b=["x", "y"])
        df2 = _df(a=[3, 4], b=["z", "w"])
        result = concatenate_dataframes([df1, df2])
        assert len(result) == 4
        assert list(result["a"]) == [1, 2, 3, 4]

    def test_ignore_index_true_by_default_resets_index(self):
        df1 = _df(a=[1, 2]).set_index(pd.Index([10, 20]))
        df2 = _df(a=[3, 4]).set_index(pd.Index([30, 40]))
        result = concatenate_dataframes([df1, df2])
        assert list(result.index) == [0, 1, 2, 3]

    def test_single_dataframe_passthrough(self):
        df = _df(a=[1, 2, 3])
        result = concatenate_dataframes([df])
        assert list(result["a"]) == [1, 2, 3]
        assert len(result) == 3

    def test_three_dataframes(self):
        dfs = [_df(a=[i]) for i in range(3)]
        result = concatenate_dataframes(dfs)
        assert list(result["a"]) == [0, 1, 2]

    def test_returns_plain_dataframe_for_plain_inputs(self):
        result = concatenate_dataframes([_df(a=[1]), _df(a=[2])])
        assert isinstance(result, pd.DataFrame)

    def test_geodataframe_concat_preserves_geodataframe_type(self, sample_trajs_gdf):
        half = len(sample_trajs_gdf) // 2
        left = sample_trajs_gdf.iloc[:half]
        right = sample_trajs_gdf.iloc[half:]
        result = concatenate_dataframes([left, right])
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(sample_trajs_gdf)
        assert list(result.columns) == list(sample_trajs_gdf.columns)


class TestAxisAndJoin:
    def test_axis_1_with_ignore_index_false_joins_side_by_side(self):
        # ignore_index resets the index along the *concatenation* axis, so
        # for axis=1 that means the resulting *column* labels get reset to
        # a plain RangeIndex unless ignore_index=False is passed explicitly.
        df1 = _df(a=[1, 2])
        df2 = _df(b=[3, 4])
        result = concatenate_dataframes([df1, df2], axis=1, ignore_index=False)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 2

    def test_axis_1_default_ignore_index_resets_column_labels(self):
        # Documents the (perhaps surprising) default behavior: because
        # `concatenate_dataframes` defaults ignore_index=True regardless of
        # axis, axis=1 concatenation silently drops the original column
        # names in favor of a 0..n-1 RangeIndex.
        df1 = _df(a=[1, 2])
        df2 = _df(b=[3, 4])
        result = concatenate_dataframes([df1, df2], axis=1)
        assert list(result.columns) == [0, 1]
        assert len(result) == 2

    def test_axis_columns_string_alias(self):
        df1 = _df(a=[1, 2])
        df2 = _df(b=[3, 4])
        result_alias = concatenate_dataframes([df1, df2], axis="columns")
        result_int = concatenate_dataframes([df1, df2], axis=1)
        pd.testing.assert_frame_equal(result_alias, result_int)

    def test_mismatched_schema_outer_join_fills_nan(self):
        df1 = _df(a=[1], b=[2])
        df2 = _df(a=[3], c=[4])
        result = concatenate_dataframes([df1, df2], join="outer")
        assert set(result.columns) == {"a", "b", "c"}
        assert result.loc[0, "b"] == 2
        assert pd.isna(result.loc[1, "b"])
        assert pd.isna(result.loc[0, "c"])
        assert result.loc[1, "c"] == 4

    def test_mismatched_schema_inner_join_keeps_common_columns_only(self):
        df1 = _df(a=[1], b=[2])
        df2 = _df(a=[3], c=[4])
        result = concatenate_dataframes([df1, df2], join="inner")
        assert set(result.columns) == {"a"}
        assert list(result["a"]) == [1, 3]

    def test_default_join_is_outer(self):
        df1 = _df(a=[1], b=[2])
        df2 = _df(a=[3])
        result = concatenate_dataframes([df1, df2])
        assert set(result.columns) == {"a", "b"}


class TestVerifyIntegrityAndSort:
    def test_verify_integrity_true_raises_on_duplicate_index(self):
        df1 = _df(a=[1]).set_index(pd.Index([0]))
        df2 = _df(a=[2]).set_index(pd.Index([0]))
        with pytest.raises(ValueError):
            concatenate_dataframes([df1, df2], ignore_index=False, verify_integrity=True)

    def test_verify_integrity_true_passes_with_ignore_index_true(self):
        # ignore_index=True (the function's default) resets the index
        # before any duplicate could be detected.
        df1 = _df(a=[1]).set_index(pd.Index([0]))
        df2 = _df(a=[2]).set_index(pd.Index([0]))
        result = concatenate_dataframes([df1, df2], verify_integrity=True)
        assert list(result.index) == [0, 1]

    def test_sort_orders_non_concat_axis_columns(self):
        df1 = _df(**{"z": [1], "a": [2]})
        df2 = _df(**{"z": [3], "a": [4], "m": [5]})
        result = concatenate_dataframes([df1, df2], sort=True)
        assert list(result.columns) == sorted(result.columns)


class TestKeysIgnoreIndexInteraction:
    def test_keys_is_a_noop_when_ignore_index_true(self):
        df1 = _df(a=[1])
        df2 = _df(a=[2])
        result = concatenate_dataframes([df1, df2], keys=["first", "second"])
        # no hierarchical index is built -- ignore_index=True wins.
        assert not isinstance(result.index, pd.MultiIndex)
        assert list(result.index) == [0, 1]

    def test_keys_builds_multiindex_when_ignore_index_false(self):
        df1 = _df(a=[1])
        df2 = _df(a=[2])
        result = concatenate_dataframes([df1, df2], keys=["first", "second"], ignore_index=False)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.get_level_values(0).tolist() == ["first", "second"]


class TestSkipSentinelFiltering:
    def test_skip_sentinel_entries_are_dropped(self):
        df1 = _df(a=[1])
        result = concatenate_dataframes([df1, SkipSentinel()])
        assert list(result["a"]) == [1]
        assert len(result) == 1

    def test_all_skip_sentinel_returns_empty_dataframe(self):
        result = concatenate_dataframes([SkipSentinel(), SkipSentinel()])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_list_returns_empty_dataframe(self):
        result = concatenate_dataframes([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_skip_sentinel_interspersed_among_real_dataframes(self):
        df1 = _df(a=[1])
        df2 = _df(a=[2])
        result = concatenate_dataframes([df1, SkipSentinel(), df2, SkipSentinel()])
        assert list(result["a"]) == [1, 2]


class TestCopyParameter:
    def test_copy_none_is_omitted_without_error(self):
        # copy=None is the default; the function should not forward it to
        # pd.concat at all (avoids the pandas 3.x deprecation warning).
        df1 = _df(a=[1])
        df2 = _df(a=[2])
        result = concatenate_dataframes([df1, df2], copy=None)
        assert list(result["a"]) == [1, 2]

    @pytest.mark.parametrize("copy_value", [True, False])
    def test_explicit_copy_value_does_not_error(self, copy_value):
        df1 = _df(a=[1])
        df2 = _df(a=[2])
        result = concatenate_dataframes([df1, df2], copy=copy_value)
        assert list(result["a"]) == [1, 2]


class TestEmptyAndSingleRowInputs:
    def test_all_empty_dataframes(self):
        df1 = pd.DataFrame({"a": pd.Series([], dtype="float64")})
        df2 = pd.DataFrame({"a": pd.Series([], dtype="float64")})
        result = concatenate_dataframes([df1, df2])
        assert result.empty
        assert "a" in result.columns

    def test_one_empty_one_non_empty(self):
        df1 = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        df2 = _df(a=[1, 2])
        result = concatenate_dataframes([df1, df2])
        assert list(result["a"]) == [1, 2]

    def test_single_row_dataframes(self):
        df1 = _df(a=[1])
        df2 = _df(a=[2])
        result = concatenate_dataframes([df1, df2])
        assert len(result) == 2
        assert list(result["a"]) == [1, 2]
