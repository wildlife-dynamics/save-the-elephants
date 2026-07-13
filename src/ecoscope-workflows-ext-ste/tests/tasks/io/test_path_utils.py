"""Tests for ecoscope_workflows_ext_ste.tasks.io._path_utils.

`get_local_file_path` is registered via `wt_registry.register()`, which is a
no-op at call time, so it behaves as a plain Python function here -- the
`Annotated[FilePath, AfterValidator(validate_any_file), ...]` signature is
not enforced by pydantic at runtime. The function body re-implements the
same checks manually (`file_path.is_file()` then `validate_any_file(...)`),
so behavior is identical whether or not pydantic validation would have run.

Suspected bug found while writing these tests (not fixed here): unlike
`_downloader.fetch_and_persist_file` / `_raster.raster_to_gdf`, which both
call `remove_file_scheme` *before* checking the path exists,
`get_local_file_path` builds `Path(file_path)` and checks `.is_file()`
*before* `remove_file_scheme` is applied (that call only happens on the
return value). So passing a `file://...` URI raises `FileNotFoundError`
even when the underlying file exists -- see
`test_file_scheme_uri_raises_file_not_found_bug` below.
"""

import os

import pytest

from ecoscope_workflows_ext_ste.tasks.io._path_utils import _ALL_FORMATS, get_local_file_path, validate_any_file


# ============================================================================
# validate_any_file
# ============================================================================


class TestValidateAnyFile:
    @pytest.mark.parametrize(
        "suffix",
        [".gpkg", ".geoparquet", ".geojson", ".csv", ".pdf", ".xlsx", ".png", ".jpg", ".json", ".parquet"],
    )
    def test_supported_extensions_pass(self, tmp_path, suffix):
        file_path = tmp_path / f"test{suffix}"
        file_path.touch()

        assert validate_any_file(file_path) == file_path

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        file_path = tmp_path / "test.exe"

        with pytest.raises(ValueError, match=r"Unsupported file format '\.exe'"):
            validate_any_file(file_path)

    def test_error_message_lists_supported_formats(self, tmp_path):
        file_path = tmp_path / "test.exe"

        with pytest.raises(ValueError, match=r"\.gpkg"):
            validate_any_file(file_path)

    def test_case_insensitive_extension(self, tmp_path):
        file_path = tmp_path / "test.GPKG"

        assert validate_any_file(file_path) == file_path

    def test_all_declared_formats_are_actually_accepted(self, tmp_path):
        for suffix in _ALL_FORMATS:
            file_path = tmp_path / f"f{suffix}"
            assert validate_any_file(file_path) == file_path


# ============================================================================
# get_local_file_path
# ============================================================================


class TestGetLocalFilePath:
    def test_resolves_real_gpkg_fixture(self, data_dir):
        path = data_dir / "kenyan_counties.gpkg"

        result = get_local_file_path(str(path))

        assert isinstance(result, str)
        assert result.endswith("kenyan_counties.gpkg")
        assert not result.startswith("file://")
        assert os.path.exists(result)

    def test_resolves_real_csv_fixture(self, data_dir):
        path = data_dir / "seasonal_windows.csv"

        result = get_local_file_path(str(path))

        assert result.endswith("seasonal_windows.csv")
        assert os.path.exists(result)

    def test_accepts_path_object(self, data_dir):
        path = data_dir / "kenyan_counties.gpkg"

        result = get_local_file_path(path)

        assert isinstance(result, str)
        assert os.path.exists(result)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist.gpkg"

        with pytest.raises(FileNotFoundError, match="File not found"):
            get_local_file_path(missing)

    def test_unsupported_extension_on_existing_file_raises_value_error(self, tmp_path):
        file_path = tmp_path / "test.exe"
        file_path.touch()

        with pytest.raises(ValueError, match="Unsupported file format"):
            get_local_file_path(file_path)

    def test_directory_is_not_a_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_local_file_path(tmp_path)

    def test_file_scheme_uri_raises_file_not_found_bug(self, tmp_path):
        """Suspected bug: `remove_file_scheme` is only applied to the return
        value, not before the `.is_file()` existence check, so a
        `file://`-prefixed path to a real, existing file is incorrectly
        reported as missing."""
        real_file = tmp_path / "test.csv"
        real_file.touch()
        uri = "file://" + str(real_file)

        with pytest.raises(FileNotFoundError):
            get_local_file_path(uri)

    def test_returns_normalized_string_without_file_scheme(self, tmp_path):
        # Sanity check: remove_file_scheme is still applied to a plain
        # (non-prefixed) resolved path -- it's simply a no-op here since
        # there was no "file://" prefix to strip in the first place.
        real_file = tmp_path / "test.csv"
        real_file.touch()

        result = get_local_file_path(str(real_file))

        assert result == str(real_file)
        assert "file://" not in result
