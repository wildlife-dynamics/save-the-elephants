"""Tests for ecoscope_workflows_ext_ste.tasks.io._downloader.

`fetch_and_persist_file` and `get_file_path` are registered via
`wt_registry.register()`, which is a no-op at call time, so they behave as
plain Python functions here -- the `Annotated[...]` signature is not enforced
by pydantic at runtime.

Unlike the pre-reorg test suite (which hit real Dropbox URLs behind
`@pytest.mark.integration`), these tests never touch the network: the
`download_file` import inside `_downloader` (from `ecoscope.io`) is patched
out everywhere, with a fake implementation that materializes a file/zip at
the target path so the surrounding path-resolution / unzip-bookkeeping logic
in `fetch_and_persist_file` still runs for real.

One suspected source bug was found while writing these tests (see comments
at the relevant test cases below) and is deliberately *not* fixed here:

1. The content-disposition filename parsing in `fetch_and_persist_file`
   builds an `email.message.Message` with the header stored under
   "content-disposition", but then calls `m.get_param("filename")` without
   `header="content-disposition"` -- `get_param` defaults to reading the
   "Content-Type" header, which was never set. So `filename` is always
   `None` and the code silently falls through to the URL-basename fallback,
   even when the server does send a real content-disposition filename.
"""

import os
import zipfile
from unittest.mock import patch

import pytest
import requests

from ecoscope_workflows_ext_ste.tasks.io._downloader import (
    DownloadFile,
    LocalFile,
    fetch_and_persist_file,
    get_file_path,
)

URL = "https://example.com/path/to/urlname.docx?rlkey=abc&dl=1"


def _make_download_file(content: bytes = b"file-bytes"):
    """A stand-in for `ecoscope.io.download_file` that just writes bytes to
    `path`, mirroring the real function's contract without any network I/O."""

    def _fake(url, path, retries, overwrite_existing, unzip):
        with open(path, "wb") as f:
            f.write(content)

    return _fake


def _make_zip_download_file(zip_arcnames: list[str], content: bytes = b"hello"):
    """A stand-in for `ecoscope.io.download_file(unzip=True)`: writes a real
    zip file to `path` and -- like the real implementation -- extracts it
    into the zip's parent directory when `unzip=True`."""

    def _fake(url, path, retries, overwrite_existing, unzip):
        with zipfile.ZipFile(path, "w") as zf:
            for name in zip_arcnames:
                zf.writestr(name, content)
        if unzip:
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(os.path.dirname(path))

    return _fake


# ============================================================================
# DownloadFile / LocalFile models
# ============================================================================


class TestDownloadFileModel:
    def test_stores_url(self):
        model = DownloadFile(url="https://example.com/f.docx")
        assert model.url == "https://example.com/f.docx"

    def test_title_config(self):
        assert DownloadFile.model_config.get("title") == "Download from URL"


class TestLocalFileModel:
    def test_stores_file_path(self):
        model = LocalFile(file_path="/some/local/path.gpkg")
        assert model.file_path == "/some/local/path.gpkg"

    def test_title_config(self):
        assert LocalFile.model_config.get("title") == "Use local file"


# ============================================================================
# fetch_and_persist_file -- output_path resolution
# ============================================================================


class TestOutputPathResolution:
    def test_none_output_path_uses_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch(
            "ecoscope_workflows_ext_ste.tasks.io._downloader.download_file",
            side_effect=_make_download_file(),
        ):
            result = fetch_and_persist_file(url=URL, output_path=None)

        assert os.path.dirname(result) == str(tmp_path)
        assert os.path.basename(result) == "urlname.docx"

    def test_empty_string_output_path_uses_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch(
            "ecoscope_workflows_ext_ste.tasks.io._downloader.download_file",
            side_effect=_make_download_file(),
        ):
            result = fetch_and_persist_file(url=URL, output_path="")

        assert os.path.dirname(result) == str(tmp_path)

    def test_whitespace_only_output_path_uses_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch(
            "ecoscope_workflows_ext_ste.tasks.io._downloader.download_file",
            side_effect=_make_download_file(),
        ):
            result = fetch_and_persist_file(url=URL, output_path="   ")

        assert os.path.dirname(result) == str(tmp_path)

    def test_explicit_full_file_path_is_used_verbatim(self, tmp_path):
        target = tmp_path / "my_template.docx"
        with patch(
            "ecoscope_workflows_ext_ste.tasks.io._downloader.download_file",
            side_effect=_make_download_file(),
        ):
            result = fetch_and_persist_file(url=URL, output_path=str(target))

        assert result == str(target.resolve())
        assert os.path.exists(result)

    def test_trailing_slash_treated_as_directory(self, tmp_path):
        dir_path = str(tmp_path) + os.sep
        with patch(
            "ecoscope_workflows_ext_ste.tasks.io._downloader.download_file",
            side_effect=_make_download_file(),
        ):
            result = fetch_and_persist_file(url=URL, output_path=dir_path)

        assert os.path.dirname(result) == str(tmp_path)
        assert os.path.basename(result) == "urlname.docx"

    def test_existing_directory_without_trailing_slash_treated_as_directory(self, tmp_path):
        with patch(
            "ecoscope_workflows_ext_ste.tasks.io._downloader.download_file",
            side_effect=_make_download_file(),
        ):
            result = fetch_and_persist_file(url=URL, output_path=str(tmp_path))

        assert os.path.dirname(result) == str(tmp_path)

    def test_directory_is_created_if_missing(self, tmp_path):
        nested = tmp_path / "nested" / "dir"
        with patch(
            "ecoscope_workflows_ext_ste.tasks.io._downloader.download_file",
            side_effect=_make_download_file(),
        ):
            result = fetch_and_persist_file(url=URL, output_path=str(nested) + os.sep)

        assert (tmp_path / "nested" / "dir").exists()
        assert os.path.exists(result)

    def test_file_scheme_prefix_is_stripped_from_output_path(self, tmp_path):
        uri = "file://" + str(tmp_path) + os.sep
        with patch(
            "ecoscope_workflows_ext_ste.tasks.io._downloader.download_file",
            side_effect=_make_download_file(),
        ):
            result = fetch_and_persist_file(url=URL, output_path=uri)

        assert os.path.dirname(result) == str(tmp_path)


# ============================================================================
# fetch_and_persist_file -- filename inference from the URL / headers
# ============================================================================


class TestFilenameInference:
    def test_falls_back_to_url_basename_when_head_request_fails(self, tmp_path):
        with (
            patch.object(requests.Session, "head", side_effect=RuntimeError("no network")),
            patch(
                "ecoscope_workflows_ext_ste.tasks.io._downloader.download_file",
                side_effect=_make_download_file(),
            ),
        ):
            result = fetch_and_persist_file(url=URL, output_path=str(tmp_path) + os.sep)

        assert os.path.basename(result) == "urlname.docx"

    def test_content_disposition_filename_is_not_actually_used(self, tmp_path):
        """Suspected bug: the server's content-disposition filename is
        ignored. `get_param("filename")` is called without
        `header="content-disposition"`, so it always reads the (never-set)
        "Content-Type" header and returns None -- the code then silently
        falls back to the URL basename, even though a real filename was
        offered by the (mocked) server."""

        class FakeResponse:
            headers = {"content-disposition": 'attachment; filename="real_name_from_server.docx"'}

        with (
            patch.object(requests.Session, "head", return_value=FakeResponse()),
            patch(
                "ecoscope_workflows_ext_ste.tasks.io._downloader.download_file",
                side_effect=_make_download_file(),
            ),
        ):
            result = fetch_and_persist_file(url=URL, output_path=str(tmp_path) + os.sep)

        # This documents *actual* current behavior, not desired behavior.
        assert os.path.basename(result) == "urlname.docx"
        assert os.path.basename(result) != "real_name_from_server.docx"

    def test_url_with_no_path_basename_falls_back_to_default_name(self, tmp_path):
        with (
            patch.object(requests.Session, "head", side_effect=RuntimeError("no network")),
            patch(
                "ecoscope_workflows_ext_ste.tasks.io._downloader.download_file",
                side_effect=_make_download_file(),
            ),
        ):
            result = fetch_and_persist_file(url="https://example.com/", output_path=str(tmp_path) + os.sep)

        assert os.path.basename(result) == "downloaded_file"


# ============================================================================
# fetch_and_persist_file -- error handling
# ============================================================================


class TestErrorHandling:
    def test_download_file_exception_is_wrapped_in_runtime_error(self, tmp_path):
        target = tmp_path / "out.docx"
        with patch(
            "ecoscope_workflows_ext_ste.tasks.io._downloader.download_file",
            side_effect=ValueError("boom"),
        ):
            with pytest.raises(RuntimeError, match="download_file failed"):
                fetch_and_persist_file(url=URL, output_path=str(target))

    def test_missing_file_after_download_raises_file_not_found_with_listing(self, tmp_path):
        target = tmp_path / "out.docx"

        def _noop(url, path, retries, overwrite_existing, unzip):
            pass  # simulate download_file silently not producing the file

        with patch("ecoscope_workflows_ext_ste.tasks.io._downloader.download_file", side_effect=_noop):
            with pytest.raises(FileNotFoundError, match="not found after execution"):
                fetch_and_persist_file(url=URL, output_path=str(target))

    def test_missing_file_and_missing_parent_dir_raises_file_not_found(self, tmp_path):
        target = tmp_path / "does_not_exist_dir" / "out.docx"

        def _noop(url, path, retries, overwrite_existing, unzip):
            pass

        with patch("ecoscope_workflows_ext_ste.tasks.io._downloader.download_file", side_effect=_noop):
            with pytest.raises(FileNotFoundError, match="Parent dir missing"):
                fetch_and_persist_file(url=URL, output_path=str(target))


# ============================================================================
# fetch_and_persist_file -- unzip bookkeeping
# ============================================================================


class TestUnzipBookkeeping:
    def test_single_extracted_directory_is_returned(self, tmp_path):
        target = tmp_path / "archive.zip"
        fake = _make_zip_download_file(["extracted_folder/file.txt"])
        with patch("ecoscope_workflows_ext_ste.tasks.io._downloader.download_file", side_effect=fake):
            result = fetch_and_persist_file(url=URL, output_path=str(target), unzip=True)

        assert result == str((tmp_path / "extracted_folder").resolve())
        assert os.path.isdir(result)

    def test_single_extracted_file_returns_parent_dir_not_the_file(self, tmp_path):
        """Quirk: when the zip's only new top-level entry is a *file* (not a
        directory), the code takes the `else` branch of the nested `if
        os.path.isdir(new_item_path)` check and returns the parent directory
        rather than the file itself."""
        target = tmp_path / "archive.zip"
        fake = _make_zip_download_file(["data.txt"])
        with patch("ecoscope_workflows_ext_ste.tasks.io._downloader.download_file", side_effect=fake):
            result = fetch_and_persist_file(url=URL, output_path=str(target), unzip=True)

        assert result == str(tmp_path.resolve())

    def test_multiple_new_items_returns_parent_dir(self, tmp_path):
        target = tmp_path / "archive.zip"
        fake = _make_zip_download_file(["a.txt", "b.txt"])
        with patch("ecoscope_workflows_ext_ste.tasks.io._downloader.download_file", side_effect=fake):
            result = fetch_and_persist_file(url=URL, output_path=str(target), unzip=True)

        assert result == str(tmp_path.resolve())

    def test_zero_new_items_falls_back_to_zip_stem_directory(self, tmp_path):
        """When the extracted top-level entry already existed in the parent
        dir *before* extraction (so it doesn't show up as a "new" item), the
        code falls back to `<zip path without .zip>` and uses that if it's a
        directory."""
        target = tmp_path / "archive.zip"
        # Pre-create the directory the zip will extract into, so it's absent
        # from `new_items` (before/after set difference cancels it out).
        (tmp_path / "archive").mkdir()
        fake = _make_zip_download_file(["archive/file.txt"])
        with patch("ecoscope_workflows_ext_ste.tasks.io._downloader.download_file", side_effect=fake):
            result = fetch_and_persist_file(url=URL, output_path=str(target), unzip=True)

        assert result == str((tmp_path / "archive").resolve())

    def test_unzip_true_but_not_a_zip_file_is_treated_as_normal_file(self, tmp_path):
        target = tmp_path / "not_a_zip.docx"
        with patch(
            "ecoscope_workflows_ext_ste.tasks.io._downloader.download_file",
            side_effect=_make_download_file(),
        ):
            result = fetch_and_persist_file(url=URL, output_path=str(target), unzip=True)

        assert result == str(target.resolve())

    def test_unzip_false_leaves_zip_file_untouched(self, tmp_path):
        target = tmp_path / "archive.zip"
        fake = _make_zip_download_file(["a.txt"])
        with patch("ecoscope_workflows_ext_ste.tasks.io._downloader.download_file", side_effect=fake):
            result = fetch_and_persist_file(url=URL, output_path=str(target), unzip=False)

        assert result == str(target.resolve())
        assert zipfile.is_zipfile(result)


# ============================================================================
# get_file_path
# ============================================================================


class TestGetFilePath:
    def test_none_input_method_returns_none(self):
        assert get_file_path(input_method=None, output_path="/some/path") is None

    def test_download_file_method_delegates_to_fetch_and_persist_file(self, tmp_path):
        download_option = DownloadFile(url=URL)
        with patch(
            "ecoscope_workflows_ext_ste.tasks.io._downloader.fetch_and_persist_file",
            return_value="/resolved/path.docx",
        ) as mock_fetch:
            result = get_file_path(input_method=download_option, output_path=str(tmp_path))

        assert result == "/resolved/path.docx"
        mock_fetch.assert_called_once_with(url=URL, output_path=str(tmp_path), unzip=False)

    def test_local_file_method_resolves_real_file(self, data_dir):
        local_option = LocalFile(file_path=str(data_dir / "kenyan_counties.gpkg"))

        result = get_file_path(input_method=local_option, output_path="/unused")

        assert result.endswith("kenyan_counties.gpkg")
        assert os.path.exists(result)

    def test_local_file_method_validates_existence(self, tmp_path):
        missing = LocalFile(file_path=str(tmp_path / "does_not_exist.gpkg"))

        with pytest.raises(FileNotFoundError):
            get_file_path(input_method=missing, output_path="/unused")

    def test_unsupported_input_method_raises_value_error(self, tmp_path):
        class UnsupportedMethod:
            pass

        with pytest.raises(ValueError, match="Unsupported input method"):
            get_file_path(input_method=UnsupportedMethod(), output_path=str(tmp_path))
