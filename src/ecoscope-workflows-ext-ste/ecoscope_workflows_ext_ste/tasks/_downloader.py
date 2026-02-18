import os
import email
import zipfile
import logging
import warnings
import requests
from pathlib import Path
from pydantic import Field
from urllib.parse import urlparse
from ecoscope.io import download_file
from typing_extensions import TypeAlias
from pydantic import BaseModel, ConfigDict
from ._path_utils import get_local_geo_path
from typing import Annotated, Union, Optional
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class DownloadFile(BaseModel):
    model_config = ConfigDict(title="Download from URL")

    url: Annotated[
        str,
        Field(
            title="URL",
            description="URL to download a file",
        ),
    ]


class LocalFile(BaseModel):
    model_config = ConfigDict(title="Use local file")

    file_path: Annotated[
        str,
        Field(
            title="Local file path",
            description="Path to a local file",
        ),
    ]


SelectPath: TypeAlias = Union[DownloadFile, LocalFile]


@task
def fetch_and_persist_file(
    url: Annotated[str, Field(description="URL to download the file from")],
    output_path: Annotated[Optional[str], Field(description="Path to save the downloaded file or directory.")] = None,
    retries: Annotated[int, Field(description="Number of retries on failure", ge=0)] = 3,
    overwrite_existing: Annotated[bool, Field(description="Whether to overwrite existing files")] = False,
    unzip: Annotated[bool, Field(description="Whether to unzip the file if it's a zip archive")] = False,
) -> str:
    """
    Downloads a file from the provided URL and persists it locally.
    If output_path is not specified, saves to the current working directory.
    Returns the full path to the downloaded file, or if unzipped, the path to the extracted directory.
    """
    logger.info(f"Downloading file from URL: {url}")
    if output_path is None or str(output_path).strip() == "":
        output_path = os.getcwd()
    else:
        output_path = str(output_path).strip()

    output_path = remove_file_scheme(output_path)
    looks_like_dir = (
        output_path.endswith(os.sep)
        or output_path.endswith("/")
        or output_path.endswith("\\")
        or os.path.isdir(output_path)
    )

    if looks_like_dir:
        os.makedirs(output_path, exist_ok=True)
        try:
            s = requests.Session()
            r = s.head(url, allow_redirects=True, timeout=10)
            cd = r.headers.get("content-disposition", "")
            filename = None
            if cd:
                # parse content-disposition safely
                m = email.message.Message()
                m["content-disposition"] = cd
                filename = m.get_param("filename")
            if not filename:
                filename = os.path.basename(urlparse(url).path.split("?")[0]) or "downloaded_file"
        except Exception:
            filename = os.path.basename(urlparse(url).path.split("?")[0]) or "downloaded_file"

        target_path = os.path.join(output_path, filename)
    else:
        target_path = output_path

    if not target_path or str(target_path).strip() == "":
        raise ValueError("Computed download target path is empty. Check 'output_path' argument.")

    parent_dir = os.path.dirname(target_path)
    before_extraction = set()
    if unzip:
        if os.path.exists(parent_dir):
            before_extraction = set(os.listdir(parent_dir))

    try:
        download_file(
            url=url,
            path=target_path,
            retries=retries,
            overwrite_existing=overwrite_existing,
            unzip=unzip,
        )
    except Exception as e:
        raise RuntimeError(
            f"download_file failed for url={url!r} path={target_path!r} retries={retries}. " f"Original error: {e}"
        ) from e

    # Determine the final persisted path
    if unzip and zipfile.is_zipfile(target_path):
        after_extraction = set(os.listdir(parent_dir))
        new_items = after_extraction - before_extraction
        zip_filename = os.path.basename(target_path)
        new_items.discard(zip_filename)

        if len(new_items) == 1:
            new_item = new_items.pop()
            new_item_path = os.path.join(parent_dir, new_item)
            if os.path.isdir(new_item_path):
                persisted_path = str(Path(new_item_path).resolve())
            else:
                persisted_path = str(Path(parent_dir).resolve())
        elif len(new_items) > 1:
            persisted_path = str(Path(parent_dir).resolve())
        else:
            extracted_dir = target_path.rsplit(".zip", 1)[0]
            if os.path.isdir(extracted_dir):
                persisted_path = str(Path(extracted_dir).resolve())
            else:
                persisted_path = str(Path(parent_dir).resolve())
    else:
        persisted_path = str(Path(target_path).resolve())

    if not os.path.exists(persisted_path):
        parent = os.path.dirname(persisted_path)
        if os.path.exists(parent):
            actual_files = os.listdir(parent)
            raise FileNotFoundError(
                f"Download failed — {persisted_path} not found after execution. " f"Files in {parent}: {actual_files}"
            )
        else:
            raise FileNotFoundError(f"Download failed — {persisted_path}. Parent dir missing: {parent}")
    logger.info(f"File downloaded and persisted at: {persisted_path}")
    return persisted_path


@task
def get_file_path(
    input_method: SelectPath,
    output_path: str,
) -> str:
    """
    Get file path based on selected input method.
    Returns the path to the (possibly extracted) file/directory ready for use.
    """
    if isinstance(input_method, DownloadFile):
        return fetch_and_persist_file(
            url=input_method.url,
            output_path=output_path,
            unzip=False,
        )
    elif isinstance(input_method, LocalFile):
        return get_local_geo_path(file_path=input_method.file_path)
    else:
        raise ValueError(f"Unsupported input method: {type(input_method)}")
