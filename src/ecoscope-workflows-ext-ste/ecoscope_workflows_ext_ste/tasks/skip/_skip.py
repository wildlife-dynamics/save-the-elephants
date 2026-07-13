from typing import Annotated, Optional
from wt_registry import register
from wt_task.skip import SkipSentinel, SKIP_SENTINEL
from ecoscope.platform.annotations import AdvancedField


@register()
def skip_toggle(
    enabled: Annotated[
        bool,
        AdvancedField(
            default=True,
            description="Shared toggle. When off, all items gated by this toggle are skipped.",
        ),
    ] = True,
) -> bool:
    """Return the enabled flag, so multiple skip_file nodes can gate on one shared toggle."""
    return enabled


@register()
def skip_file(
    filename: Annotated[
        Optional[str],
        AdvancedField(default="animated.html", description="Animated html file."),
    ] = "animated.html",
    enabled: Annotated[
        bool,
        AdvancedField(
            default=False,
            description="Render the animated map as a video file. "
            "When off, the video creation step is skipped entirely.",
        ),
    ] = False,
) -> Optional[str] | SkipSentinel:
    """Return the output filename, or a skip sentinel when video export is disabled."""
    if not enabled:
        return SKIP_SENTINEL
    return filename
