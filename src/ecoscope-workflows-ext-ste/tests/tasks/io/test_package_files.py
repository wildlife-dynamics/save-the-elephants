"""Tests for ecoscope_workflows_ext_ste.tasks.io._package_files."""

from pathlib import Path

import pytest

from ecoscope_workflows_ext_ste.tasks.io import MAPBOOK_FILES, load_mapbook_file


@pytest.mark.parametrize("filename", MAPBOOK_FILES)
def test_every_packaged_file_resolves_to_a_local_file(filename):
    path = Path(load_mapbook_file(filename))

    assert path.is_absolute()
    assert path.is_file()
    assert path.name == filename


def test_unknown_file_raises():
    with pytest.raises(ValueError, match="Unknown Mapbook file"):
        load_mapbook_file("not_packaged.docx")
