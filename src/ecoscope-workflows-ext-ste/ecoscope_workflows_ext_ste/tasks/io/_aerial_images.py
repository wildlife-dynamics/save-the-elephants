import pandas as pd
import datetime as dt
from pathlib import Path
import concurrent.futures
from pydantic import Field
from tqdm.auto import tqdm
from typing import Annotated
from wt_registry import register
from PIL import Image as PilImage
from pydantic.json_schema import SkipJsonSchema
from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.tasks.filter import TimezoneInfo
from ecoscope.platform.connections import EarthRangerClient

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp"}


# Standard TIFF/EXIF tag IDs — Pillow's _getexif() returns a dict keyed by these integers.
# Values are defined by the EXIF spec and are the same across all camera manufacturers.
# fmt: off
_EXIF_DATETIME_ORIGINAL = 36867
_EXIF_DATETIME          = 306
_EXIF_MAKE              = 271
_EXIF_MODEL             = 272
# fmt: on


def _read_exif(path: Path) -> dict:
    """Extract timestamp and camera metadata from a single image via Pillow EXIF."""
    record = {
        "file_name": path.name,
        "file_path": str(path),
        "datetime": None,
        "make": None,
        "model": None,
    }
    try:
        img = PilImage.open(path)
        exif = img._getexif()
        if not exif:
            return record
        raw_dt = exif.get(_EXIF_DATETIME_ORIGINAL) or exif.get(_EXIF_DATETIME)
        record["datetime"] = dt.datetime.strptime(raw_dt, "%Y:%m:%d %H:%M:%S") if raw_dt else None
        record["make"] = exif.get(_EXIF_MAKE)
        record["model"] = exif.get(_EXIF_MODEL)
    except Exception as exc:
        print(f"  [warn] EXIF read failed for {path.name}: {exc}")
    return record


@register()
def process_aerial_images(
    image_folder: Annotated[str, Field(description="Path to the folder containing aerial survey images.")],
    timezone: Annotated[
        TimezoneInfo | SkipJsonSchema[None],
        Field(description="Timezone the camera clock was set to when images were captured."),
    ] = None,
) -> AnyDataFrame:
    """
    Scan an image folder, extract EXIF timestamps and camera metadata.

    Walks the folder recursively, reads EXIF data from every supported image,
    and localises all datetimes to the camera's timezone so they can be matched
    against EarthRanger event times in the same timezone.

    Returns one row per image: file_name, file_path, datetime, make, model.
    Images without readable EXIF timestamps are included with datetime=NaT
    and will be excluded during the matching step.
    """
    folder = Path(image_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    print(f"Scanning images in: {folder}")
    records = [_read_exif(p) for p in sorted(folder.rglob("*")) if p.suffix.lower() in _IMAGE_SUFFIXES]
    if not records:
        raise ValueError(f"No supported images found in: {folder}")

    tz = timezone.utc_offset if timezone is not None else "UTC"

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.tz_localize(tz)

    n_valid = df["datetime"].notna().sum()
    print(f"Processed {len(df)} images — {n_valid} with readable timestamps, {len(df) - n_valid} skipped.")
    return df


@register()
def upload_images_to_er_events(
    client: Annotated[EarthRangerClient, Field(description="EarthRanger client from set_er_connection")],
    matched_df: Annotated[AnyDataFrame, Field(description="Output of match_images_to_events")],
) -> AnyDataFrame:
    """
    Upload matched images as file attachments to their EarthRanger events.

    POSTs each image to /api/v1.0/activity/event/{id}/files/ using the authenticated
    ER session. Uploads run concurrently up to the client's TCP limit. Every attempt
    — success or failure — is recorded in the returned DataFrame:
    event_id, serial_number, file_name, status_code, success, uploaded_at.
    """
    uploads = [
        (row["event_id"], row["serial_number"], file_path)
        for _, row in matched_df.iterrows()
        for file_path in row["matched_images"]
    ]

    def _upload(event_id, serial, file_path):
        path = Path(file_path)
        record = {
            "event_id": event_id,
            "serial_number": serial,
            "file_name": path.name,
            "success": False,
            "uploaded_at": pd.Timestamp.now("UTC"),
        }
        try:
            client.post_event_file(event_id=event_id, filepath=str(path))
            record["success"] = True
        except Exception as exc:
            print(f"  [error] {path.name} → event {serial}: {exc}")
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=client.tcp_limit) as executor:
        futures = [executor.submit(_upload, *args) for args in uploads]

    results = [
        future.result()
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Uploading images",
        )
    ]
    uploaded = sum(r["success"] for r in results)
    print(f"Upload complete: {uploaded} / {len(results)} files succeeded.")
    return pd.DataFrame(results)
