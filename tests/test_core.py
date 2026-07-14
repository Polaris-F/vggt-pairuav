from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from pairuav.geometry import d_from_frames, frame_step, grid_from_d_values, labels_from_d
from pairuav.metrics import compute_metrics_np
from pairuav.postproc_maphard import default_weights_path, map_decode


ROOT = Path(__file__).resolve().parents[1]


class CoreMathTests(unittest.TestCase):
    def test_public_training_recipe_is_pinned(self) -> None:
        submission_dir = ROOT / "configs/submission"
        lamp_dir = ROOT / "configs/lamp"
        submission_angle = json.loads((submission_dir / "angle_s0.json").read_text(encoding="utf-8"))
        lamp_angle = json.loads((lamp_dir / "angle_s0.json").read_text(encoding="utf-8"))
        submission_close = json.loads((submission_dir / "range_c_rel_rich.json").read_text(encoding="utf-8"))
        lamp_range = json.loads((lamp_dir / "range_ab_relsmooth.json").read_text(encoding="utf-8"))
        submission_system = json.loads((submission_dir / "system.json").read_text(encoding="utf-8"))
        lamp_system = json.loads((lamp_dir / "system.json").read_text(encoding="utf-8"))

        self.assertEqual(submission_angle, lamp_angle)
        self.assertEqual(lamp_angle["epochs"], 90)
        self.assertEqual(submission_close["input_mode"], "ab_diff_prod")
        self.assertEqual(submission_close["lr"], 2e-3)
        self.assertEqual(submission_close["epochs"], 120)
        self.assertEqual(lamp_range["input_mode"], "ab")
        self.assertEqual(lamp_range["loss"], "rel_smooth")
        self.assertEqual(lamp_range["lr"], 1e-3)
        self.assertEqual(lamp_range["epochs"], 240)
        self.assertEqual(lamp_range["batch_size"], 512)
        self.assertEqual(submission_system["gate_threshold_m"], 80.0)
        self.assertIsNotNone(submission_system["range2_config"])
        self.assertIsNone(lamp_system["range2_config"])

    def test_geometry_mapping(self) -> None:
        self.assertEqual(frame_step(1), 0)
        self.assertEqual(frame_step(2), 4)
        self.assertEqual(frame_step(54), 264)
        self.assertEqual(d_from_frames(1, 54), 264)
        heading, range_m = labels_from_d(np.array([0, 4, 264, -264]))
        np.testing.assert_allclose(heading, [0, 16, -24, 24])
        np.testing.assert_allclose(range_m, [0, -2, -132, 132])

    def test_map_grid_decodes_to_itself(self) -> None:
        grid_d, grid_h, grid_r = grid_from_d_values()
        weights = json.loads(default_weights_path().read_text(encoding="utf-8"))
        decoded = map_decode(
            grid_h,
            grid_r,
            grid_d,
            grid_h,
            grid_r,
            float(weights["w_h"]),
            float(weights["w_r"]),
        )
        self.assertEqual(len(grid_d), 211)
        np.testing.assert_array_equal(decoded, grid_d)

    def test_perfect_predictions_have_zero_error(self) -> None:
        heading = np.array([0.0, 16.0, -24.0])
        range_m = np.array([0.0, -2.0, -132.0])
        metrics = compute_metrics_np(heading, range_m, heading, range_m)
        self.assertEqual(metrics["final_score"], 0.0)
        self.assertEqual(metrics["skipped_final_samples"], 0)

    def test_index_manifest_hashes(self) -> None:
        manifest = json.loads((ROOT / "data_index/manifest.json").read_text(encoding="utf-8"))
        for split in manifest["splits"]:
            lines = [
                line.strip()
                for line in (ROOT / "data_index" / split["file"]).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            ]
            digest = hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
            self.assertEqual(len(lines), split["count"], split["name"])
            self.assertEqual(len(set(lines)), split["unique_count"], split["name"])
            self.assertEqual(len(lines) - len(set(lines)), split["duplicate_rows"], split["name"])
            if not split["duplicates_allowed"]:
                self.assertEqual(len(lines), len(set(lines)), split["name"])
            self.assertEqual(digest, split["sha256"], split["name"])


if __name__ == "__main__":
    unittest.main()
