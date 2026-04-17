import concurrent.futures
import datetime as dt
from pathlib import Path
from typing import Annotated, List

from PIL import Image as PilImage
from pydantic.json_schema import SkipJsonSchema

import pandas as pd
from pydantic import Field
from tqdm.auto import tqdm
from ecoscope_workflows_core.annotations import AnyDataFrame
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.filter import TimezoneInfo
from ecoscope_workflows_ext_ecoscope.connections import EarthRangerClient

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


def _match_window(
    event_time: pd.Timestamp,
    images_df: pd.DataFrame,
    window: pd.Timedelta,
    created_at: pd.Timestamp | None = None,
) -> List[str]:
    """Return file paths for images captured within the match window.

    Lower bound is event_time - window. Upper bound is created_at + window when
    created_at > event_time (ER Mobile: event started, photos taken, then submitted),
    otherwise event_time + window.
    """
    upper = (created_at if created_at and created_at > event_time else event_time) + window
    mask = (images_df["datetime"] >= event_time - window) & (images_df["datetime"] <= upper)
    return images_df.loc[mask, "file_path"].tolist()


@task
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


@task
def match_images_to_events(
    images_df: Annotated[AnyDataFrame, Field(description="Output of process_aerial_images")],
    events_df: Annotated[AnyDataFrame, Field(description="EarthRanger events DataFrame")],
    time_window_minutes: Annotated[
        float,
        Field(description="Time window in minutes for matching images to events.", gt=0),
    ] = 4.0,
) -> AnyDataFrame:
    """
    Assign aerial images to EarthRanger events by timestamp proximity.

    For each event, every image captured within ±time_window_minutes of the
    event time is collected. Only events with at least one matched image appear
    in the output, which is intended as the input to both build_match_preview_table
    and upload_images_to_er_events.
    """
    window = pd.Timedelta(minutes=time_window_minutes)
    timestamped = images_df.dropna(subset=["datetime"]).copy()

    print(
        f"Matching {len(timestamped)} timestamped images against "
        f"{len(events_df)} events (\u00b1{time_window_minutes} min window)..."
    )

    rows = []
    for _, event in events_df.iterrows():
        event_time = pd.to_datetime(event["time"])
        if event_time.tzinfo is None:
            event_time = event_time.tz_localize("UTC")

        raw_created_at = event.get("created_at")
        created_at = pd.to_datetime(raw_created_at, utc=True) if raw_created_at else None

        matched = _match_window(event_time, timestamped, window, created_at)
        if matched:
            rows.append(
                {
                    "event_id": event.get("id", event.name),
                    "serial_number": event.get("serial_number"),
                    "event_time": event_time,
                    "created_at": created_at,
                    "event_type": event.get("event_type"),
                    "event_type_display": event.get("event_type_display"),
                    "matched_images": matched,
                    "image_count": len(matched),
                }
            )

    result = pd.DataFrame(rows)
    total_images = int(result["image_count"].sum()) if not result.empty else 0
    print(f"Matched {len(result)} of {len(events_df)} events — {total_images} images queued for upload.")
    return result


@task
def get_unmatched_images(
    images_df: Annotated[AnyDataFrame, Field(description="Output of process_aerial_images")],
    matched_df: Annotated[AnyDataFrame, Field(description="Output of match_images_to_events")],
) -> AnyDataFrame:
    """
    Return images that were not matched to any EarthRanger event.

    Diffs the full scanned image set against the matched paths collected
    in match_images_to_events. Only timestamped images are considered —
    images without readable EXIF are excluded from both matched and unmatched.
    """
    matched_paths = set(p for paths in matched_df["matched_images"] for p in paths)
    timestamped = images_df.dropna(subset=["datetime"])
    return timestamped[~timestamped["file_path"].isin(matched_paths)].reset_index(drop=True)


_CSS = """
<style>
  .aerial { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: .875rem; color: #1a1a1a; }
  .aerial .scroll-wrap { overflow-y: auto; max-height: 480px; }
  .aerial table { border-collapse: collapse; width: 100%; }
  .aerial th { background: #475569; color: #fff; font-weight: 600; padding: 10px 14px;
               text-align: left; font-size: .875rem; letter-spacing: .03em;
               text-transform: uppercase; position: sticky; top: 0; z-index: 1; }
  .aerial td { padding: 9px 14px; border-bottom: 1px solid #e8e8e8;
               vertical-align: top; font-size: .875rem; }
  .aerial tr:last-child td { border-bottom: none; }
  .aerial tbody tr:nth-child(odd) { background: #f9f9fb; }
  .aerial tbody tr:hover { background: #eef2ff; }
  .aerial .files { font-size: .875rem; color: #555; line-height: 1.8; }
  .aerial .badge { display: inline-block; background: #e8f0fe; color: #1a56db;
                   border-radius: 12px; padding: 1px 9px; font-size: .78rem; font-weight: 600; }
  .aerial .summary { margin-top: 10px; color: #666; font-size: .82rem; display: flex; gap: 18px; }
  .aerial .summary span { display: flex; align-items: center; gap: 5px; }
  .aerial .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .aerial .dot-green { background: #22c55e; }
  .aerial .dot-grey  { background: #94a3b8; }
</style>
"""


@task
def build_matched_table(
    matched_df: Annotated[AnyDataFrame, Field(description="Output of match_images_to_events")],
) -> str:
    """
    Render a styled HTML table of events with their matched images.

    Shows event serial number, time, type, image count, and filenames.
    Includes a summary line with total events matched and images queued.
    """
    if matched_df.empty:
        return f"{_CSS}<p class='aerial' style='color:#888;padding:12px 0'>No images matched any events.</p>"

    def _row(r: pd.Series) -> str:
        filenames = "<br>".join(Path(p).name for p in r["matched_images"])
        return (
            f"<tr>"
            f"<td>{r['serial_number']}</td>"
            f"<td>{r['event_time']}</td>"
            f"<td>{r.get('event_type_display') or r['event_type']}</td>"
            f"<td><span class='badge'>{r['image_count']}</span></td>"
            f"<td class='files'>{filenames}</td>"
            f"</tr>"
        )

    body = "".join(_row(r) for _, r in matched_df.iterrows())
    table = (
        "<table>"
        "<thead><tr>"
        "<th>Event #</th><th>Event Time</th><th>Event Type</th><th># Images</th><th>Image Files</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )
    summary = (
        "<div class='summary'>"
        f"<span><span class='dot dot-green'></span> {len(matched_df)} events matched</span>"
        f"<span><span class='dot dot-green'></span> {int(matched_df['image_count'].sum())} images queued</span>"
        "</div>"
    )
    return f"{_CSS}<div class='aerial'><div class='scroll-wrap'>{table}</div>{summary}</div>"


@task
def build_unmatched_table(
    unmatched_df: Annotated[AnyDataFrame, Field(description="Output of get_unmatched_images")],
) -> str:
    """
    Render a styled HTML table of images that were not matched to any event.

    Shows filename and EXIF timestamp for each unmatched image.
    """
    if unmatched_df.empty:
        return f"{_CSS}<p class='aerial' style='color:#888;padding:12px 0'>All images were matched to events.</p>"

    # fmt: off
    rows = "".join(
        f"<tr><td>{r['file_name']}</td><td>{r['datetime']}</td></tr>"
        for _, r in unmatched_df.iterrows()
    )
    table = (
        "<table>"
        "<thead><tr><th>Filename</th><th>Timestamp</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )
    # fmt: on
    summary = (
        "<div class='summary'>"
        f"<span><span class='dot dot-grey'></span> {len(unmatched_df)} images unmatched</span>"
        "</div>"
    )
    return f"{_CSS}<div class='aerial'><div class='scroll-wrap'>{table}</div>{summary}</div>"


@task
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
