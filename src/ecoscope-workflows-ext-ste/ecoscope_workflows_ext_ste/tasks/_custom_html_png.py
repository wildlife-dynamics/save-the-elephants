import re
import tempfile
from pathlib import Path
from typing import Union, Optional
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_ext_custom.tasks.io._html_to_png import ScreenshotConfig, html_to_png


@task
def adjust_map_zoom_and_screenshot(
    input_file: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = None,
    zoom_value: float = 1.05,
    screenshot_config: Optional[ScreenshotConfig] = None,
) -> Optional[str]:
    """
    Adjust the zoom level in a pydeck HTML map file, take a screenshot, and clean up.

    This function:
    1. Reads the HTML file
    2. Adjusts the zoom level
    3. Saves to a temporary file
    4. Takes a screenshot using html_to_png with the same filename as input
    5. Removes the temporary file
    6. Returns the path to the PNG image

    Args:
        input_file: Path to the input HTML file (can be None, will return None)
        output_dir: Directory to save the PNG screenshot
        zoom_value: Zoom value to replace
        screenshot_config: Optional ScreenshotConfig for html_to_png

    Returns:
        Path to the generated PNG file, or None if input_file is None
    """
    # Handle None input gracefully - skip processing
    if input_file is None:
        print("Input_file is None - skipping processing")
        return None

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}\n" f"Check that the HTML generation task completed successfully."
        )

    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    print("\n=== Processing HTML file ===")
    print(f"Input file: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Target zoom: {zoom_value}")

    # Read the HTML file
    with open(input_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Find and adjust the zoom value using regex
    zoom_pattern = r'"zoom":\s*(\d+\.?\d*)'
    match = re.search(zoom_pattern, html_content)

    if not match:
        raise ValueError(
            f"Could not find zoom value in the HTML file: {input_path}\n"
            f"The file may not be a valid pydeck HTML map."
        )

    current_zoom = float(match.group(1))
    print(f"Current zoom level: {current_zoom}")
    print(f"New zoom value: {zoom_value}")

    # Replace the zoom value
    new_html_content = re.sub(zoom_pattern, f'"zoom": {zoom_value}', html_content)

    # Extract the base filename (without extension) from input file
    original_filename = input_path.stem  # Gets filename without extension

    # Create a temporary file with the adjusted HTML
    # Use the original filename for the temp file to preserve it for the PNG
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", prefix=f"{original_filename}_", delete=False, encoding="utf-8"
    ) as temp_file:
        temp_file.write(new_html_content)
        temp_html_path = temp_file.name

    print(f"Created temporary file: {temp_html_path}")

    try:
        # Use the existing html_to_png function to take screenshot
        if screenshot_config is None:
            screenshot_config = ScreenshotConfig()

        # Call html_to_png which will generate the PNG
        png_path = html_to_png(html_path=temp_html_path, output_dir=str(output_dir), config=screenshot_config)

        # Rename the PNG to match the original HTML filename
        png_path_obj = Path(png_path)
        desired_png_name = f"{original_filename}.png"
        desired_png_path = png_path_obj.parent / desired_png_name

        # If the generated PNG has a different name, rename it
        if png_path_obj.name != desired_png_name:
            png_path_obj.rename(desired_png_path)
            png_path = str(desired_png_path)
            print(f"Renamed PNG to: {desired_png_name}")

        print(f"\n✓ Successfully saved screenshot to: {png_path}")
        print(f"✓ Zoom changed from {current_zoom} to {zoom_value}")

        return png_path

    finally:
        # Clean up temporary file
        temp_path = Path(temp_html_path)
        if temp_path.exists():
            temp_path.unlink()
            print(f"✓ Temporary file removed: {temp_html_path}")
