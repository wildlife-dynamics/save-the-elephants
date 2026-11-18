"""Checks on the packaged mapbook Word templates and the report built from them.

`merge_docx_documents` keeps only the cover page's headers and footers, so the
"Powered by Ecoscope" footer has to live in the cover template to reach the
merged report.
"""

import hashlib
import re
import zipfile
from datetime import datetime, timezone

from docxtpl import DocxTemplate

from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange
from ecoscope_workflows_ext_ste.tasks.io._package_files import load_mapbook_file
from ecoscope_workflows_ext_ste.tasks.reporting._cover_context import create_context_page, prepare_cover_metadata
from ecoscope_workflows_ext_ste.tasks.reporting._merge_documents import merge_docx_documents


def _footers(docx_path: str) -> dict[str, str]:
    """Footer XML of the document's last section, by type (default/first/even)."""
    with zipfile.ZipFile(docx_path) as z:
        doc = z.read("word/document.xml").decode()
        rels = z.read("word/_rels/document.xml.rels").decode()
        targets = {}
        for rel in re.findall(r"<Relationship [^>]*>", rels):
            rel_id = re.search(r'Id="([^"]+)"', rel)
            target = re.search(r'Target="([^"]+)"', rel)
            if rel_id and target:
                targets[rel_id.group(1)] = target.group(1)
        sect = re.findall(r"<w:sectPr.*?</w:sectPr>", doc, flags=re.S)[-1]
        return {
            kind: z.read("word/" + targets[rel_id]).decode()
            for kind, rel_id in re.findall(r'<w:footerReference w:type="(\w+)" r:id="(\w+)"', sect)
        }


def _media_hashes(docx_path: str) -> set[str]:
    with zipfile.ZipFile(docx_path) as z:
        return {hashlib.md5(z.read(n)).hexdigest() for n in z.namelist() if n.startswith("word/media/")}


def _build_report(tmp_path) -> tuple[str, str]:
    time_range = TimeRange(
        since=datetime(2026, 8, 1, tzinfo=timezone.utc),
        until=datetime(2026, 11, 9, tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )
    # The mapbook spec passes the Wildlife Dynamics logo path to the cover page.
    context = prepare_cover_metadata(
        org_logo_path=load_mapbook_file("wd-logo.png"),
        report_period=time_range,
        prepared_by="Ecoscope",
        extra_fields={"subject_count": 2},
    )
    cover = create_context_page(
        template_path=load_mapbook_file("mapbook_cover_page.docx"),
        output_dir=str(tmp_path),
        context=context,
        filename="cover_page.docx",
    )
    pages = []
    for name in ("subject_a", "subject_b"):
        page = DocxTemplate(load_mapbook_file("mapbook_grouper_template.docx"))
        page.render({"grouper_value": name})
        path = str(tmp_path / f"{name}.docx")
        page.save(path)
        pages.append(path)
    merged = merge_docx_documents(
        cover_page_path=cover,
        context_page_items=pages,
        output_dir=str(tmp_path),
        filename="overall_report.docx",
    )
    return cover.replace("file://", ""), merged.replace("file://", "")


def test_report_has_no_wildlife_dynamics_logo(tmp_path):
    """No logo on the report until STE decide whether one should be shown."""
    cover, merged = _build_report(tmp_path)
    with open(load_mapbook_file("wd-logo.png"), "rb") as f:
        wd_logo = hashlib.md5(f.read()).hexdigest()

    assert wd_logo not in _media_hashes(cover)
    assert wd_logo not in _media_hashes(merged)


def test_merged_report_has_powered_by_ecoscope_footer(tmp_path):
    _, merged = _build_report(tmp_path)
    footers = _footers(merged)

    assert "Powered by" in footers["default"]
    assert "r:embed" in footers["default"]  # the Ecoscope logo image
    # The cover is a title page with its own, empty footer.
    assert "Powered by" not in footers["first"]
