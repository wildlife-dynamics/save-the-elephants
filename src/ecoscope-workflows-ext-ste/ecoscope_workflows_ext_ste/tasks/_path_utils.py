from pathlib import Path
from typing import Annotated
from ecoscope_workflows_core.decorators import task
from pydantic import Field, FilePath, AfterValidator
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme


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
    file_path_str = str(file_path)
    normalized_path = remove_file_scheme(file_path_str)
    return normalized_path
