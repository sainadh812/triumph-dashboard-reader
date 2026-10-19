"""Offline geometry, packaging and video IO checks. No Torch or neural execution."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

from triumph_dashboard_reader import inference, video, weights
from triumph_dashboard_reader.contracts import ARCHITECTURE_ID, PREPROCESSING_ID
from triumph_dashboard_reader.release import WEIGHTS_SHA256


class GeometryTests(unittest.TestCase):
    def test_reference_tick_returns_exact_known_speed(self):
        reference = inference.SPEED_REFERENCE
        points = np.asarray(
            [reference["pivot"], reference["ticks"][3], *reference["ticks"]]
        )
        result = inference.read_speed(points, np.ones(12))
        self.assertIsNone(result["reason"])
        self.assertAlmostEqual(result["value_kmh"], 60, places=4)
        self.assertFalse(result["diagnostics"]["physical_accuracy_verified"])

    def test_rpm_interpolation_and_invisible_endpoint(self):
        points = np.asarray([[50, 20 + i * 40] for i in range(7)] + [[50, 80]], float)
        self.assertAlmostEqual(inference.read_rpm(points, np.ones(8))["value"], 3000)
        visibility = np.ones(8)
        visibility[7] = 0
        self.assertEqual(
            inference.read_rpm(points, visibility)["reason"],
            "rpm_fill_endpoint_not_visible",
        )

    def test_selects_single_lcd_component_inside_dashboard(self):
        probability = np.zeros((50, 60))
        probability[10:20, 10:20] = 1
        probability[25:30, 10:16] = 1
        probability[2:48, 40:58] = 1  # Larger false region outside dashboard.
        mask, polygon = inference.select_lcd_mask(probability, [5, 5, 30, 40])
        self.assertEqual(np.count_nonzero(mask), 100)
        self.assertEqual(len(polygon), 4)

    def test_heatmap_pixel_centers_and_projective_horizon(self):
        logits = np.zeros((1, 1, 2, 2))
        logits[0, 0, 1, 1] = 1
        np.testing.assert_allclose(
            inference.decode_heatmaps(logits, (8, 8)), [[[5.5, 5.5]]]
        )
        with self.assertRaisesRegex(ValueError, "projective_horizon"):
            inference.project_points([[0, 0]], [[1, 0, 0], [0, 1, 0], [1, 0, 0]])

    def test_local_execution_rejected_before_weight_loading(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(inference.importlib.util, "find_spec", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "restricted to Google Colab"):
                inference.DashboardPredictor({})


class VideoTests(unittest.TestCase):
    def test_full_video_default_and_final_window(self):
        metadata = {
            "timestamps": [0.0, 0.25, 0.5, 0.75],
            "last_frame_period_seconds": 0.25,
        }
        self.assertEqual(video.clip_indices(metadata, None), (0, 4, 0.0, 1.0))
        self.assertEqual(video.clip_indices(metadata, 0.5), (2, 4, 0.5, 1.0))
        with self.assertRaises(ValueError):
            video.clip_indices(metadata, 0)

    def test_conditional_coverage_does_not_confuse_rejection_and_skip(self):
        prediction = {
            "speed": {"point_visibility": {"pivot": 0.8}, "value_kmh": None},
            "rpm": {"point_visibility": {}},
        }
        self.assertEqual(
            video._task_inference_flags(prediction),
            {"localizer": True, "speed": True, "rpm": False},
        )

    def test_padding_preserves_every_original_pixel_and_coordinates(self):
        original = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        padded = video.pad_for_encoder(original)
        self.assertEqual(padded.shape, (6, 8, 3))
        np.testing.assert_array_equal(padded[:5, :7], original)
        self.assertFalse(padded[5].any())
        self.assertFalse(padded[:, 7].any())

    def test_rpm_overlay_font_larger_than_speed(self):
        calls = []

        def record(image, text, position, face, scale, *args):
            calls.append((text, scale))
            return image

        with patch.object(cv2, "putText", side_effect=record):
            output = video.draw_prediction(
                np.zeros((700, 900, 3), np.uint8),
                {"speed": {"value_kmh": 40}, "rpm": {"value": 3000}},
                0,
                "test",
            )
        speeds = [scale for text, scale in calls if text.startswith("Speed:")]
        rpms = [scale for text, scale in calls if text.startswith("RPM:")]
        self.assertGreater(min(rpms), max(speeds))
        self.assertEqual(output.shape, (700, 900, 3))

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg/ffprobe required for video IO test",
    )
    def test_full_pipeline_with_fake_predictions_and_odd_dimensions(self):
        class FakePredictor:
            def predict_frames(self, frames):
                return [
                    {
                        "speed": {
                            "value_kmh": None,
                            "reason": "test_skip",
                            "point_visibility": {},
                        },
                        "rpm": {
                            "value": None,
                            "reason": "test_skip",
                            "point_visibility": {},
                        },
                    }
                    for _ in frames
                ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "odd.mkv"
            frames = np.full((4, 49, 65, 3), 100, np.uint8)
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    "65x49",
                    "-r",
                    "4",
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "ffv1",
                    str(source),
                ],
                input=frames.tobytes(),
                check=True,
                capture_output=True,
            )
            report = video.process_video(
                source, root / "out", FakePredictor(), batch_size=3
            )
            self.assertEqual(report["frames_processed"], 4)
            self.assertEqual(report["original_decoded_height_width"], [49, 65])
            self.assertEqual(report["encoded_height_width"], [50, 66])
            self.assertEqual(
                report["actual_task_inference_frames"],
                {"localizer": 4, "speed": 0, "rpm": 0, "all_three": 0},
            )
            self.assertEqual(report["unavailable_reasons"]["speed"], {"test_skip": 4})
            self.assertTrue((root / "out" / "labeled.mp4").is_file())
            self.assertFalse((root / "out" / "labeled_intermediate.mp4").exists())


class PackagingTests(unittest.TestCase):
    def make_archive(self, path, *, bad_digest=False, extra=None):
        payloads = {
            f"{task}/best_model.pt": task.encode()
            for task in ("localizer", "speed", "rpm")
        }
        manifest = {
            "schema_version": 1,
            "release_tag": "v0.1.0",
            "architecture_id": ARCHITECTURE_ID,
            "preprocessing_id": PREPROCESSING_ID,
            "dataset_manifest_sha256": "a" * 64,
            "selected_epochs": {"localizer": 5, "speed": 0, "rpm": 14},
            "files": [
                {
                    "path": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                }
                for name, data in payloads.items()
            ],
        }
        if bad_digest:
            manifest["files"][0]["sha256"] = "0" * 64
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("weights_manifest.json", json.dumps(manifest))
            for name, data in payloads.items():
                archive.writestr(name, data)
            if extra:
                archive.writestr(extra, "unexpected")
        return weights.sha256(path), payloads

    def test_verified_extraction_reads_no_neural_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digest, payloads = self.make_archive(root / "weights.zip")
            paths, manifest = weights.extract_verified(
                root / "weights.zip", root / "models", digest
            )
            for task, path in paths.items():
                self.assertEqual(path.read_bytes(), payloads[f"{task}/best_model.pt"])
            self.assertEqual(manifest["selected_epochs"]["speed"], 0)

    def test_member_hash_failure_writes_no_partial_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digest, _ = self.make_archive(root / "weights.zip", bad_digest=True)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                weights.extract_verified(root / "weights.zip", root / "models", digest)
            self.assertFalse((root / "models").exists())

    def test_unexpected_or_traversal_member_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digest, _ = self.make_archive(root / "weights.zip", extra="../outside.txt")
            with self.assertRaisesRegex(ValueError, "archive members"):
                weights.extract_verified(root / "weights.zip", root / "models", digest)
            self.assertFalse((root / "outside.txt").exists())

    def test_release_hash_and_notebook_syntax(self):
        self.assertEqual(
            WEIGHTS_SHA256,
            "2ae6c5539fa097154fd57eec984bdeaffecd92d0f67fe18e6e9747c86d585be5",
        )
        notebook = json.loads(
            (
                Path(__file__).parents[1] / "notebooks" / "Test_Your_Video.ipynb"
            ).read_text()
        )
        sources = {
            cell.get("id"): "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        }
        for name, source in sources.items():
            compile(source, name, "exec")
        self.assertIn('CODE_REF + "^{commit}"', sources["setup"])
        self.assertIn("DOWNLOADS_CONFIRMED_SAVED = False", sources["disconnect"])
        self.assertNotIn(
            "runtime.unassign",
            "".join(source for name, source in sources.items() if name != "disconnect"),
        )
        self.assertFalse(any(cell.get("outputs") for cell in notebook["cells"]))


if __name__ == "__main__":
    unittest.main()
