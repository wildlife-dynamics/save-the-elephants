"""Progress lines for the STE workflows.

Every task prints through `log`, so a run's log reads as the stages of the
workflow, each line saying what was done and with how much data:

    [STE Tracks] segment filter: kept 5,828 of 6,010 segments
    [STE Seasons] NDVI from Earth Engine (14 requests): done in 18.4s

The stages are not numbered: these tasks are shared by the mapbook, the general
report and the aerial survey workflows, which run them in different orders.
"""

import time
from contextlib import contextmanager
from typing import Iterator

STAGES = {
    "setup": "Setup",
    "files": "Files",
    "tracks": "Tracks",
    "ranges": "Home range",
    "seasons": "Seasons",
    "maps": "Maps",
    "tables": "Tables",
    "report": "Report",
    "aerial": "Aerial survey",
}


def log(stage: str, message: str) -> None:
    """Print one progress line under a stage (a key of STAGES)."""
    print(f"[STE {STAGES.get(stage, stage)}] {message}", flush=True)


def count(n: int, noun: str, plural: str | None = None) -> str:
    """'1 segment', '1,556 segments'; pass `plural` for nouns that do not just add an s."""
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {plural or noun + 's'}"


@contextmanager
def timed(stage: str, what: str) -> Iterator[None]:
    """Print when a step starts and how long it took, also when it fails."""
    log(stage, f"{what}...")
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        log(
            stage,
            f"{what}: failed after {time.perf_counter() - start:.1f}s ({type(exc).__name__})",
        )
        raise
    log(stage, f"{what}: done in {time.perf_counter() - start:.1f}s")
