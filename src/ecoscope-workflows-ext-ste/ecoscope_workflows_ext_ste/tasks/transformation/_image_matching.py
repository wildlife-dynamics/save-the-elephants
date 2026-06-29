import pandas as pd
from pydantic import Field
from wt_registry import register
from typing import Annotated, List
from ecoscope.platform.annotations import AnyDataFrame


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


@register()
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


@register()
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
