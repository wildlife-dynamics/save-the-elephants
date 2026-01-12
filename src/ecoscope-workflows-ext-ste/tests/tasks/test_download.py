import pytest
import os
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from ecoscope_workflows_ext_ste.tasks._downloader import (
    fetch_and_persist_file,
    get_file_path,
    DownloadFile,
    LocalFile,
)

# Test URLs
DROPBOX_DOCX_URL = "https://www.dropbox.com/scl/fi/s85tmsn4ed5es18xkykw9/grouper_template.docx?rlkey=wdtzx9ry51fxncgoakeydit3l&st=0upmlflg&dl=1"
TEST_SHAPEFILE_URL = "https://www.dropbox.com/scl/fi/phlc488gxqpcvr6ua3vk7/amboseli_group_ranch_boundaries.gpkg?rlkey=p5ztypwmj4ndjova9xe2ssiun&st=7yywvjni&dl=0"


@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory."""
    with TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def sample_zip_file(temp_dir):
    """Fixture to create a sample zip file."""
    zip_path = os.path.join(temp_dir, "test_archive.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Create a test file inside the zip
        test_content = "Test file content"
        zf.writestr("test_file.txt", test_content)
    return zip_path


@pytest.fixture
def kenyan_counties_path():
    """Fixture to get the path to the Kenyan counties file."""
    TEST_DATA_DIR = Path(__file__).parent.parent / "data"
    return str(TEST_DATA_DIR / "kenyan_counties.gpkg")


# Tests for fetch_and_persist_file
@pytest.mark.integration
def test_download_docx_from_dropbox(temp_dir):
    """Test downloading a real .docx file from Dropbox."""
    result = fetch_and_persist_file(
        url=DROPBOX_DOCX_URL, output_path=temp_dir, retries=3, overwrite_existing=False, unzip=False
    )

    assert os.path.exists(result)
    assert result.endswith(".docx")
    assert os.path.getsize(result) > 0
    assert os.path.dirname(result) == str(Path(temp_dir).resolve())


@pytest.mark.integration
def test_download_to_specific_filename(temp_dir):
    """Test downloading to a specific filename."""
    target_file = os.path.join(temp_dir, "my_template.docx")

    result = fetch_and_persist_file(url=DROPBOX_DOCX_URL, output_path=target_file, retries=3)

    assert os.path.exists(result)
    assert result.endswith("my_template.docx")
    assert os.path.getsize(result) > 0


def test_download_to_default_directory():
    """Test downloading with no output_path specified (uses cwd)."""
    result = fetch_and_persist_file(
        url="https://www.dropbox.com/scl/fi/1373gi65ji918rxele5h9/cover_page_v3.docx?rlkey=ur01wtpa98tcyq8f0f6dtksl8&st=3iwbmp7y&dl=0",
        output_path=None,
    )

    # Should use current working directory
    assert os.path.dirname(result) == os.getcwd()


# Overwrite test
@pytest.mark.integration
def test_download_overwrite_existing_file(temp_dir):
    """Test overwriting an existing file."""
    # First download
    result1 = fetch_and_persist_file(url=DROPBOX_DOCX_URL, output_path=temp_dir, overwrite_existing=False)

    original_size = os.path.getsize(result1)

    # Second download with overwrite
    result2 = fetch_and_persist_file(url=DROPBOX_DOCX_URL, output_path=temp_dir, overwrite_existing=True)

    assert result1 == result2
    assert os.path.exists(result2)
    # File should be re-downloaded
    assert os.path.getsize(result2) == original_size


# Error handling tests
def test_empty_string_output_path_uses_cwd(temp_dir, monkeypatch):
    monkeypatch.chdir(temp_dir)  # optional - make assertion easier

    result = fetch_and_persist_file(
        url=DROPBOX_DOCX_URL,
        output_path="",  # empty string
    )

    assert os.path.dirname(result) == os.getcwd()


# Tests for get_file_path
@pytest.mark.integration
def test_download_file_method_real_download(temp_dir):
    """Test get_file_path with real download from Dropbox."""
    download_option = DownloadFile(url=DROPBOX_DOCX_URL)

    result = get_file_path(input_method=download_option, output_path=temp_dir)

    assert os.path.exists(result)
    assert result.endswith(".docx")
    assert os.path.getsize(result) > 0


def test_local_file_method(kenyan_counties_path):
    """Test get_file_path with LocalFile method."""
    local_option = LocalFile(file_path=kenyan_counties_path)

    result = get_file_path(
        input_method=local_option,
        output_path="/dummy/path",  # Not used for local files
    )

    assert os.path.exists(result)
    assert result.endswith("kenyan_counties.gpkg")


def test_local_file_method_validates_path(temp_dir):
    """Test that LocalFile validates the file path."""
    # Create a temporary file
    test_file = os.path.join(temp_dir, "test.gpkg")
    Path(test_file).touch()

    local_option = LocalFile(file_path=test_file)

    result = get_file_path(input_method=local_option, output_path="/dummy/path")

    assert os.path.exists(result)
    assert "test.gpkg" in result


def test_unsupported_method_raises_error(temp_dir):
    """Test that unsupported input method raises ValueError."""

    class UnsupportedMethod:
        pass

    with pytest.raises(ValueError, match="Unsupported input method"):
        get_file_path(input_method=UnsupportedMethod(), output_path=temp_dir)


def test_download_file_pydantic_validation():
    """Test that DownloadFile validates URL format."""
    # Valid URL
    valid = DownloadFile(
        url="https://www.dropbox.com/scl/fi/1373gi65ji918rxele5h9/cover_page_v3.docx?rlkey=ur01wtpa98tcyq8f0f6dtksl8&st=3iwbmp7y&dl=0"
    )
    assert (
        valid.url
        == "https://www.dropbox.com/scl/fi/1373gi65ji918rxele5h9/cover_page_v3.docx?rlkey=ur01wtpa98tcyq8f0f6dtksl8&st=3iwbmp7y&dl=0"
    )

    # Pydantic should accept any string for url field
    another_valid = DownloadFile(
        url="https://www.dropbox.com/scl/fi/1373gi65ji918rxele5h9/cover_page_v3.docx?rlkey=ur01wtpa98tcyq8f0f6dtksl8&st=3iwbmp7y&dl=0"
    )
    assert (
        another_valid.url
        == "https://www.dropbox.com/scl/fi/1373gi65ji918rxele5h9/cover_page_v3.docx?rlkey=ur01wtpa98tcyq8f0f6dtksl8&st=3iwbmp7y&dl=0"
    )


def test_local_file_pydantic_validation(kenyan_counties_path):
    """Test that LocalFile validates file path."""
    valid = LocalFile(file_path=kenyan_counties_path)
    assert valid.file_path == kenyan_counties_path
