from pydantic import Field
from wt_registry import register
from wt_task.skip import SKIP_SENTINEL
from typing import Optional, Any, Annotated, Literal
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme


@register()
def merge_docx_documents(
    cover_page_path: Annotated[str, Field(description="Path to the cover page .docx file")],
    context_page_items: Annotated[list[Any], Field(description="List of context page document paths to merge.")],
    output_dir: Annotated[str, Field(description="Directory where combined docx will be written")],
    filename: Annotated[Optional[str], Field(description="Optional output filename")] = None,
    order_by: Annotated[
        Literal["name", "month", "input"],
        Field(description="Page ordering: alphabetical by filename, calendar month, or input order"),
    ] = "name",
) -> Annotated[str, Field(description="Path to the combined .docx file")]:
    """Combine cover + context pages into a single DOCX."""
    import calendar
    import os
    import re
    from datetime import datetime
    from pathlib import Path

    from docx import Document
    from docxcompose.composer import Composer

    cover_page_path = remove_file_scheme(cover_page_path)
    if not os.path.exists(cover_page_path):
        raise FileNotFoundError(f"Cover page file not found: {cover_page_path}")

    output_dir = remove_file_scheme(output_dir)
    if not output_dir.strip():
        raise ValueError("output_dir is empty after normalization")
    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        filename = f"overall_report_{datetime.now():%Y%m%d_%H%M%S}.docx"
    elif not filename.lower().endswith(".docx"):
        filename = f"{filename}.docx"
    output_path = Path(output_dir) / filename

    # ---------- normalize page items ----------
    def extract_path(item) -> Optional[str]:
        if item is None or item is SKIP_SENTINEL:
            return None
        if isinstance(item, str):
            return remove_file_scheme(item)
        if isinstance(item, (list, tuple)):
            for x in item:
                if x is SKIP_SENTINEL:
                    return None
            for x in item:
                if isinstance(x, str) and os.path.exists(remove_file_scheme(x)):
                    return remove_file_scheme(x)
        return None

    pages = []
    for idx, item in enumerate(context_page_items):
        path = extract_path(item)
        if path is None:
            print(f"Skipping page item {idx}: no usable path ({item!r})")
        elif not os.path.exists(path):
            print(f"Skipping page item {idx}: file not found: {path}")
        else:
            pages.append(path)

    if order_by == "name":
        pages.sort(key=lambda p: os.path.basename(p).lower())
    elif order_by == "month":
        month_words = {
            name.lower(): i
            for names in (calendar.month_name, calendar.month_abbr)
            for i, name in enumerate(names)
            if name
        }
        pattern = re.compile(r"(?<![a-z])(" + "|".join(sorted(month_words, key=len, reverse=True)) + r")(?![a-z])")

        def month_key(path: str):
            m = pattern.search(os.path.basename(path).lower())
            return (0, month_words[m.group(1)]) if m else (1, 0)

        pages.sort(key=month_key)
    master = Document(cover_page_path)
    composer = Composer(master)

    if not pages:
        print("No valid context pages to merge; saving cover page only")
    else:
        for page_path in pages:
            composer.append(Document(page_path))
    composer.save(str(output_path))
    print(f"Merged {len(pages)} pages into {output_path}")
    return str(output_path)
