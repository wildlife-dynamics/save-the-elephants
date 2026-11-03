from pathlib import Path
from pydantic import Field
from wt_registry import register
from typing import Annotated, Literal, get_args
from importlib.resources import as_file, files


MapbookFile = Literal[
    "mapbook_cover_page.docx", "mapbook_grouper_template.docx", "mapbook_ldx_layers.gpkg", "wd-logo.png"
]

MAPBOOK_FILES: tuple[str, ...] = get_args(MapbookFile)


@register()
def load_mapbook_file(
    filename: Annotated[MapbookFile, Field(description="Name of the mapbook file to load.")],
) -> Annotated[str, Field(description="Local path to the packaged file.")]:
    """Return the local path of a file shipped with ext-ste.

    Replaces downloading map layers and report templates at run time: the file is
    read from the installed package, so it works the same on desktop and on
    ecoscope-web, without network access.
    """
    if filename not in MAPBOOK_FILES:
        raise ValueError(f"Unknown Mapbook file {filename!r}. Available: {', '.join(MAPBOOK_FILES)}")
    resource = files("ecoscope_workflows_ext_ste") / "data" / filename
    # Installed packages are plain folders, so as_file yields the real file path.
    with as_file(resource) as path:
        local = Path(path)
    if not local.is_file():
        raise FileNotFoundError(f"Packaged file is missing: {local}")
    return str(local.resolve())
