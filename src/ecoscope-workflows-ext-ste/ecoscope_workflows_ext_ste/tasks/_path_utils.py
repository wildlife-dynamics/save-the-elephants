import os
from pathlib import Path
from typing import Annotated
from ecoscope_workflows_core.decorators import task
from pydantic import Field, FilePath, AfterValidator


def normalize_file_url(path: str) -> str:
    """Convert file:// URL to local path, handling malformed Windows URLs."""
    if not path.startswith("file://"):
        return path

    path = path[7:]

    if os.name == "nt":
        # Remove leading slash before drive letter: /C:/path -> C:/path
        if path.startswith("/") and len(path) > 2 and path[2] in (":", "|"):
            path = path[1:]

        path = path.replace("/", "\\")
        path = path.replace("|", ":")
    else:
        if not path.startswith("/"):
            path = "/" + path

    return path


def validate_geo_file(path: Path) -> Path:
    valid_formats = [".shp", ".gpkg", ".geoparquet"]
    if path.suffix.lower() not in valid_formats:
        raise ValueError(f"Invalid file format '{path.suffix}'. Allowed formats are: {', '.join(valid_formats)}")
    return path


@task
def get_local_geo_path(
    file_path: Annotated[
        FilePath,
        AfterValidator(validate_geo_file),
        Field(description="Path to the geospatial file (shapefile, geopackage, or geoparquet)."),
    ],
) -> str:
    # FilePath already validates the file exists, so we can normalize it
    file_path_str = str(file_path)
    normalized_path = normalize_file_url(file_path_str)
    return normalized_path
