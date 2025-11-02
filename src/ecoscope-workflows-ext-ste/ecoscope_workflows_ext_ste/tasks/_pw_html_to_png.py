from typing import Annotated

from ecoscope_workflows_core.decorators import task
from pydantic import BaseModel, Field


class ScreenshotConfig(BaseModel):
    width: int = 1280
    height: int = 720
    full_page: bool = False
    device_scale_factor: float = 2.0
    wait_for_timeout: int = 30_000


@task
def html_to_png_pw(
    html_path: Annotated[str, Field(description="The html file path")],
    output_dir: Annotated[str, Field(description="The output root path")],
    config: Annotated[
        ScreenshotConfig, Field(description="The screenshot configuration")
    ] = ScreenshotConfig(),
) -> str:
    from pathlib import Path

    from playwright.sync_api import sync_playwright

    png_filename = Path(html_path).with_suffix(".png").name

    # todo: handle this more gracefully
    if output_dir.startswith("file://"):
        # Remove 'file://' prefix to get the base path
        output_dir = output_dir[7:]

    output_path = Path(output_dir) / png_filename

    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": config.width, "height": config.height},
            device_scale_factor=config.device_scale_factor,
        )

        page.goto(Path(html_path).as_uri())

        # Wait for network activity to settle
        page.wait_for_load_state("networkidle", timeout=0)
        page.wait_for_timeout(config.wait_for_timeout)

        # Take screenshot
        page.screenshot(path=output_path, full_page=True, timeout=0)

        # Close browser
        browser.close()
    return str(output_path)