import pandas as pd
from pathlib import Path
from pydantic import Field
from typing import Annotated
from wt_registry import register
from ecoscope.platform.annotations import AnyDataFrame

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


@register()
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


@register()
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
