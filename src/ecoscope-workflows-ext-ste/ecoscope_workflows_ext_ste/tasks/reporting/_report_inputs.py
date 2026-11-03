"""Shared helpers for the report tasks: find their input files and read them.

The report tasks take their images and tables as explicit paths, which may be
local paths, ``file://`` URLs or GCS paths (desktop and ecoscope-web). Without
explicit paths they fall back to scanning a local results folder, as before.
"""

import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from ecoscope_workflows_ext_custom.storage import as_readable, is_gcs, normalize_uri
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}


def paths_only(items: Optional[list[Any]]) -> list[str]:
    """Flatten report inputs to a list of paths.

    Accepts plain paths, and the grouped results a ``mapvalues`` step returns:
    ``(group_key, path)`` pairs, or ``(group_key, [paths])`` when a step
    returns several files per group. Empty values are dropped.
    """
    out: list[str] = []

    def add(item: Any) -> None:
        if not item:
            return
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, tuple) and len(item) == 2 and not isinstance(item[0], str):
            add(item[1])  # (group_key, value)
        elif isinstance(item, (list, tuple)):
            for x in item:
                add(x)

    for item in items or []:
        add(item)
    return out


def file_stem(path: str) -> str:
    """File name without extension, for local paths, file:// and GCS paths alike.

    GCS read paths are URL-encoded (``ESELENKEI%20GR.png``), so decode them first.
    """
    return Path(normalize_uri(path).rstrip("/").split("/")[-1]).stem


def file_suffix(path: str) -> str:
    return Path(normalize_uri(path)).suffix.lower()


def scan_local_dir(output_dir: str, task_name: str) -> list[str]:
    """Legacy fallback: list the files in a local results folder.

    GCS folders can't be scanned; callers must pass their input files instead.
    """
    if is_gcs(output_dir):
        raise ValueError(
            f"{task_name}: pass the input files explicitly when output_dir is on GCS; "
            "scanning a GCS folder is not supported."
        )
    local_dir = remove_file_scheme(output_dir)
    os.makedirs(local_dir, exist_ok=True)  # as before: the folder exists even if nothing is found
    found = []
    for root, _, files in os.walk(local_dir):
        found.extend(str(Path(root) / f) for f in files)
    return found


def by_stem(paths: list[str], suffixes: set[str]) -> dict[str, str]:
    """Map file stem -> path, for the paths with one of the given suffixes."""
    return {file_stem(p): p for p in paths if p and file_suffix(p) in suffixes}


def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(as_readable(path))


def find(name: str, paths: list[str]) -> Optional[str]:
    """Exact stem match first, then a case-insensitive substring match."""
    for p in paths:
        if file_stem(p) == name:
            return p
    for p in paths:
        if name.lower() in file_stem(p).lower():
            return p
    return None
