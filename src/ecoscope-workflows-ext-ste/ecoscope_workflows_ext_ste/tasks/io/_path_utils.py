from pathlib import Path
from typing import Annotated
from wt_registry import register
from pydantic import Field, FilePath, AfterValidator
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

_ALL_FORMATS = {
    # Geospatial
    ".gpkg",
    ".geoparquet",
    ".geojson",
    ".kml",
    ".kmz",
    ".glb",
    ".gltf",
    ".obj",
    ".fbx",
    ".docx",
    ".pdf",
    ".xlsx",
    ".csv",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".svg",
    ".json",
    ".parquet",
    ".feather",
    ".nc",
    ".h5",
    ".hdf5",
}


def validate_any_file(path: Path) -> Path:
    if path.suffix.lower() not in _ALL_FORMATS:
        raise ValueError(
            f"Unsupported file format '{path.suffix}'. Supported formats are: {', '.join(sorted(_ALL_FORMATS))}"
        )
    return path


@register()
def get_local_file_path(
    file_path: Annotated[
        FilePath,
        AfterValidator(validate_any_file),
        Field(
            description=(
                "Path to any supported file. Accepted formats: "
                "geospatial (.gpkg, .geoparquet, .geojson,.kml, .kmz), "
                "3D (.glb, .gltf, .obj, .fbx), "
                "documents (.docx, .pdf, .xlsx, .csv, .txt), "
                "images/rasters (.png, .jpg, .jpeg, .tif, .tiff, .svg), "
                "data (.json, .parquet, .feather, .nc, .h5, .hdf5)."
            )
        ),
    ],
) -> str:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    validate_any_file(file_path)
    print(f"[get_local_file_path] Resolved path: {file_path} (type: {file_path.suffix.lower()})")
    return remove_file_scheme(str(file_path))
