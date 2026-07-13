"""Tests for ecoscope_workflows_ext_ste.tasks.io._aerial_images.

Both `process_aerial_images` and `upload_images_to_er_events` are registered
via `wt_registry.register()`, which is a no-op at call time, so they behave
as plain Python functions here.

`process_aerial_images` reads real EXIF metadata from image files on disk
via Pillow -- these tests build small real JPEGs with `PIL.Image` (writing
EXIF tags directly with `Image.getexif()`, no `piexif` dependency needed) so
the actual EXIF-parsing logic in `_read_exif` is exercised for real.

`upload_images_to_er_events` talks to EarthRanger via
`client.post_event_file(...)`. That client is typed as
`ecoscope.platform.connections.EarthRangerClient`, an `Annotated` protocol
alias -- since `@register()` doesn't enforce it at call time, a plain
`unittest.mock.MagicMock` (with `tcp_limit` set to a real int, since it's
used as `ThreadPoolExecutor(max_workers=...)`) stands in for a real client.
"""

import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from PIL import Image

from ecoscope.platform.tasks.filter._filter import TimezoneInfo
from ecoscope_workflows_ext_ste.tasks.io._aerial_images import process_aerial_images, upload_images_to_er_events

# EXIF tag IDs (see _aerial_images.py for the same constants).
_EXIF_DATETIME_ORIGINAL = 36867
_EXIF_MAKE = 271
_EXIF_MODEL = 272


def _make_image(
    path: Path,
    datetime_original: str | None = None,
    make: str | None = None,
    model: str | None = None,
) -> Path:
    img = Image.new("RGB", (4, 4), color="red")
    if datetime_original or make or model:
        exif = img.getexif()
        if datetime_original:
            exif[_EXIF_DATETIME_ORIGINAL] = datetime_original
        if make:
            exif[_EXIF_MAKE] = make
        if model:
            exif[_EXIF_MODEL] = model
        img.save(path, exif=exif)
    else:
        img.save(path)
    return path


# ============================================================================
# process_aerial_images
# ============================================================================


class TestProcessAerialImages:
    def test_reads_exif_datetime_make_and_model(self, tmp_path):
        _make_image(
            tmp_path / "img1.jpg",
            datetime_original="2023:06:15 08:30:00",
            make="Canon",
            model="EOS R5",
        )

        result = process_aerial_images(image_folder=str(tmp_path))

        assert len(result) == 1
        row = result.iloc[0]
        assert row["file_name"] == "img1.jpg"
        assert row["make"] == "Canon"
        assert row["model"] == "EOS R5"
        assert row["datetime"] == pd.Timestamp("2023-06-15 08:30:00", tz="UTC")

    def test_default_timezone_is_utc(self, tmp_path):
        _make_image(tmp_path / "img.jpg", datetime_original="2023:06:15 08:30:00")

        result = process_aerial_images(image_folder=str(tmp_path))

        assert str(result.iloc[0]["datetime"].tz) == "UTC"

    def test_custom_timezone_is_applied(self, tmp_path):
        _make_image(tmp_path / "img.jpg", datetime_original="2023:06:15 08:30:00")
        tz = TimezoneInfo(label="EAT", tzCode="Africa/Nairobi", name="East Africa Time", utc_offset="+03:00")

        result = process_aerial_images(image_folder=str(tmp_path), timezone=tz)

        localized = result.iloc[0]["datetime"]
        assert localized.utcoffset() == dt.timedelta(hours=3)
        # tz_localize (not tz_convert) is used, so the naive wall-clock time
        # is unchanged -- it is just labeled as being in the new timezone.
        assert localized.hour == 8

    def test_image_without_exif_gets_nat_datetime(self, tmp_path):
        _make_image(tmp_path / "no_exif.jpg")

        result = process_aerial_images(image_folder=str(tmp_path))

        assert len(result) == 1
        assert pd.isna(result.iloc[0]["datetime"])
        assert result.iloc[0]["make"] is None
        assert result.iloc[0]["model"] is None

    def test_mixed_images_with_and_without_exif(self, tmp_path):
        _make_image(tmp_path / "a_with_exif.jpg", datetime_original="2023:01:01 00:00:00")
        _make_image(tmp_path / "b_no_exif.jpg")

        result = process_aerial_images(image_folder=str(tmp_path))

        assert len(result) == 2
        assert result["datetime"].notna().sum() == 1
        assert result["datetime"].isna().sum() == 1

    def test_non_image_files_are_ignored(self, tmp_path):
        _make_image(tmp_path / "img.jpg", datetime_original="2023:01:01 00:00:00")
        (tmp_path / "notes.txt").write_text("not an image")
        (tmp_path / "data.csv").write_text("a,b\n1,2")

        result = process_aerial_images(image_folder=str(tmp_path))

        assert len(result) == 1
        assert result.iloc[0]["file_name"] == "img.jpg"

    def test_recurses_into_subfolders(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        _make_image(tmp_path / "top.jpg", datetime_original="2023:01:01 00:00:00")
        _make_image(sub / "nested.jpg", datetime_original="2023:01:02 00:00:00")

        result = process_aerial_images(image_folder=str(tmp_path))

        assert set(result["file_name"]) == {"top.jpg", "nested.jpg"}

    @pytest.mark.parametrize("suffix", [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"])
    def test_all_declared_image_suffixes_are_scanned(self, tmp_path, suffix):
        path = tmp_path / f"img{suffix}"
        img = Image.new("RGB", (4, 4))
        img.save(path)

        result = process_aerial_images(image_folder=str(tmp_path))

        assert len(result) == 1
        assert result.iloc[0]["file_name"] == f"img{suffix}"

    def test_missing_folder_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist"

        with pytest.raises(FileNotFoundError, match="Image folder not found"):
            process_aerial_images(image_folder=str(missing))

    def test_empty_folder_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="No supported images found"):
            process_aerial_images(image_folder=str(tmp_path))

    def test_folder_with_only_non_image_files_raises_value_error(self, tmp_path):
        (tmp_path / "readme.txt").write_text("hello")

        with pytest.raises(ValueError, match="No supported images found"):
            process_aerial_images(image_folder=str(tmp_path))

    def test_result_has_expected_columns(self, tmp_path):
        _make_image(tmp_path / "img.jpg", datetime_original="2023:01:01 00:00:00")

        result = process_aerial_images(image_folder=str(tmp_path))

        assert list(result.columns) == ["file_name", "file_path", "datetime", "make", "model"]


# ============================================================================
# upload_images_to_er_events
# ============================================================================


@pytest.fixture
def er_client():
    client = MagicMock()
    client.tcp_limit = 2
    return client


class TestUploadImagesToErEvents:
    def test_uploads_all_images_and_reports_success(self, er_client, tmp_path):
        img_a = tmp_path / "a.jpg"
        img_b = tmp_path / "b.jpg"
        img_a.touch()
        img_b.touch()
        matched_df = pd.DataFrame(
            {
                "event_id": ["e1"],
                "serial_number": ["s1"],
                "matched_images": [[str(img_a), str(img_b)]],
            }
        )

        result = upload_images_to_er_events(client=er_client, matched_df=matched_df)

        assert er_client.post_event_file.call_count == 2
        assert len(result) == 2
        assert result["success"].all()
        assert set(result["file_name"]) == {"a.jpg", "b.jpg"}
        assert list(result.columns) == ["event_id", "serial_number", "file_name", "success", "uploaded_at"]

    def test_multiple_rows_each_with_multiple_images(self, er_client, tmp_path):
        images = [tmp_path / f"img{i}.jpg" for i in range(4)]
        for p in images:
            p.touch()
        matched_df = pd.DataFrame(
            {
                "event_id": ["e1", "e2"],
                "serial_number": ["s1", "s2"],
                "matched_images": [
                    [str(images[0]), str(images[1])],
                    [str(images[2]), str(images[3])],
                ],
            }
        )

        result = upload_images_to_er_events(client=er_client, matched_df=matched_df)

        assert len(result) == 4
        assert er_client.post_event_file.call_count == 4

    def test_calls_post_event_file_with_event_id_and_filepath(self, er_client, tmp_path):
        img = tmp_path / "only.jpg"
        img.touch()
        matched_df = pd.DataFrame({"event_id": ["e42"], "serial_number": ["s7"], "matched_images": [[str(img)]]})

        upload_images_to_er_events(client=er_client, matched_df=matched_df)

        er_client.post_event_file.assert_called_once_with(event_id="e42", filepath=str(img))

    def test_partial_failure_is_recorded_per_image(self, er_client, tmp_path):
        good = tmp_path / "good.jpg"
        bad = tmp_path / "bad.jpg"
        good.touch()
        bad.touch()

        def _side_effect(event_id, filepath):
            if filepath == str(bad):
                raise RuntimeError("upload failed")

        er_client.post_event_file.side_effect = _side_effect
        matched_df = pd.DataFrame(
            {"event_id": ["e1"], "serial_number": ["s1"], "matched_images": [[str(good), str(bad)]]}
        )

        result = upload_images_to_er_events(client=er_client, matched_df=matched_df)

        by_name = result.set_index("file_name")
        assert by_name.loc["good.jpg", "success"]
        assert not by_name.loc["bad.jpg", "success"]

    def test_all_failures_returns_dataframe_with_success_false(self, er_client, tmp_path):
        img = tmp_path / "img.jpg"
        img.touch()
        er_client.post_event_file.side_effect = RuntimeError("network down")
        matched_df = pd.DataFrame({"event_id": ["e1"], "serial_number": ["s1"], "matched_images": [[str(img)]]})

        result = upload_images_to_er_events(client=er_client, matched_df=matched_df)

        assert len(result) == 1
        assert not result.iloc[0]["success"]

    def test_empty_matched_df_returns_empty_result(self, er_client):
        matched_df = pd.DataFrame(columns=["event_id", "serial_number", "matched_images"])

        result = upload_images_to_er_events(client=er_client, matched_df=matched_df)

        assert len(result) == 0
        er_client.post_event_file.assert_not_called()

    def test_row_with_empty_matched_images_list_contributes_no_uploads(self, er_client, tmp_path):
        img = tmp_path / "img.jpg"
        img.touch()
        matched_df = pd.DataFrame(
            {
                "event_id": ["e1", "e2"],
                "serial_number": ["s1", "s2"],
                "matched_images": [[], [str(img)]],
            }
        )

        result = upload_images_to_er_events(client=er_client, matched_df=matched_df)

        assert len(result) == 1
        assert result.iloc[0]["file_name"] == "img.jpg"
