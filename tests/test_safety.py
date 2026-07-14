from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from pairuav.cache import slice_cache
from pairuav.gate_merge import main as gate_merge_main
from pairuav.index import command_materialize, command_verify_cache
from pairuav.postproc_maphard import write_submission


class SafetyTests(unittest.TestCase):
    def test_workflow_scripts_have_valid_bash_syntax(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for script in (root / "scripts/reproduce_submission.sh", root / "scripts/train_lamp.sh"):
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_cache_slice_rejects_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            out = root / "out"
            source.mkdir()
            out.mkdir()
            np.save(source / "features.npy", np.zeros((2, 2, 4), dtype=np.float16))
            np.save(source / "heading.npy", np.zeros(2, dtype=np.float32))
            np.save(source / "range.npy", np.zeros(2, dtype=np.float32))
            (source / "json_paths.json").write_text('["a", "b"]\n', encoding="utf-8")
            (source / "meta.json").write_text('{"samples": 2}\n', encoding="utf-8")
            (out / "keep.txt").write_text("keep\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                slice_cache(source, out, limit=1)
            with self.assertRaises(ValueError):
                slice_cache(source, root / "negative", limit=1, offset=-1)

    def test_batched_pair_input_matches_full_input(self) -> None:
        try:
            import torch
            from pairuav.heads import make_pair_input
        except Exception as exc:  # pragma: no cover - environment-dependent skip
            self.skipTest(f"PyTorch stack unavailable: {exc}")

        generator = torch.Generator().manual_seed(2026)
        features = torch.randn((7, 2, 16), generator=generator, dtype=torch.float16)
        for mode in ("ab", "ab_diff_prod", "diff_prod"):
            full = make_pair_input(features, mode)
            batched = torch.cat([make_pair_input(features[start: start + 3], mode) for start in range(0, 7, 3)])
            torch.testing.assert_close(batched, full, rtol=0.0, atol=0.0)

    def test_training_config_requires_all_fields(self) -> None:
        try:
            from pairuav.train_range import load_config
        except Exception as exc:  # pragma: no cover - environment-dependent skip
            self.skipTest(f"PyTorch stack unavailable: {exc}")

        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "incomplete.json"
            config.write_text(json.dumps({"name": "incomplete", "lr": 1e-3}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(config)

    def test_training_run_directory_must_be_empty(self) -> None:
        try:
            from pairuav.reproducibility import prepare_run_dir
        except Exception as exc:  # pragma: no cover - environment-dependent skip
            self.skipTest(f"PyTorch stack unavailable: {exc}")

        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            prepare_run_dir(run_dir)
            (run_dir / "config.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_run_dir(run_dir)

    def test_materialize_rejects_stale_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            out = root / "out"
            index = root / "index.txt"
            for rel in ("0001/01_02.json", "0001/02_03.json"):
                path = source / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            index.write_text("0001/01_02.json\n0001/02_03.json\n", encoding="utf-8")
            stale = out / "0001/03_04.json"
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_text("{}\n", encoding="utf-8")
            args = argparse.Namespace(
                index=index,
                source_json_dir=source,
                out_json_dir=out,
                name="test",
            )
            with self.assertRaises(RuntimeError):
                command_materialize(args)

    def test_verify_cache_rejects_wrong_array_length(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = root / "index.txt"
            cache = root / "cache"
            cache.mkdir()
            lines = ["0001/01_02.json", "0001/02_03.json"]
            index.write_text("\n".join(lines) + "\n", encoding="utf-8")
            (cache / "json_paths.json").write_text(json.dumps(lines), encoding="utf-8")
            (cache / "meta.json").write_text(json.dumps({"samples": 2}), encoding="utf-8")
            np.save(cache / "features.npy", np.zeros((1, 2, 4), dtype=np.float16))
            np.save(cache / "heading.npy", np.zeros(2, dtype=np.float32))
            np.save(cache / "range.npy", np.zeros(2, dtype=np.float32))
            args = argparse.Namespace(index=index, cache_dir=cache)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                command_verify_cache(args)

            np.save(cache / "features.npy", np.zeros((2, 2, 4), dtype=np.float16))
            with contextlib.redirect_stdout(io.StringIO()):
                command_verify_cache(args)

    def test_gate_merge_requires_complete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base.txt"
            shard = root / "shard.txt"
            out = root / "out.txt"
            np.savetxt(base, np.array([[0, 10], [4, 90], [8, 100]], dtype=float))
            np.savetxt(shard, np.array([[0, 10, 11], [4, 90, 91]], dtype=float))
            argv = [
                "gate_merge",
                "--base-result", str(base),
                "--pred", f"{shard}:0",
                "--out", str(out),
            ]
            with mock.patch.object(sys, "argv", argv), self.assertRaises(RuntimeError):
                gate_merge_main()
            self.assertFalse(out.exists())

            np.savetxt(shard, np.array([[40, 10, 11], [4, 90, 91], [8, 100, 101]], dtype=float))
            with mock.patch.object(sys, "argv", argv), self.assertRaises(ValueError):
                gate_merge_main()
            self.assertFalse(out.exists())

            np.savetxt(shard, np.array([[0, 10, 11], [4, 90, 91], [8, 100, 101]], dtype=float))
            with mock.patch.object(sys, "argv", argv):
                gate_merge_main()
            self.assertTrue(out.is_file())

    def test_maphard_writer_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "submission"
            heading = np.array([0.0, 4.0])
            range_m = np.array([0.0, -0.5])
            write_submission(out, heading, range_m, expected_lines=2)
            with self.assertRaises(FileExistsError):
                write_submission(out, heading, range_m, expected_lines=2)

    def test_angle_training_always_writes_best_checkpoints(self) -> None:
        try:
            import torch
            from pairuav.train_angle import AngleConfig, train_one
        except Exception as exc:  # pragma: no cover - environment-dependent skip
            self.skipTest(f"PyTorch training stack unavailable: {exc}")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_cache = root / "train"
            val_cache = root / "val"
            train_cache.mkdir()
            val_cache.mkdir()
            features = np.zeros((4, 2, 2), dtype=np.float32)
            np.save(train_cache / "features.npy", features)
            np.save(val_cache / "features.npy", features)
            geometry = {
                "heading": np.zeros(4, dtype=np.float32),
                "range": np.zeros(4, dtype=np.float32),
                "rot": np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0),
                "trans_world": np.zeros((4, 3), dtype=np.float32),
            }
            train_geom = root / "train_geom.npz"
            val_geom = root / "val_geom.npz"
            np.savez(train_geom, **geometry)
            np.savez(val_geom, **geometry)
            args = argparse.Namespace(
                train_cache=train_cache,
                val_cache=val_cache,
                train_geom=train_geom,
                val_geom=val_geom,
                eval_batch_size=4,
                seed=2026,
            )
            cfg = AngleConfig(
                name="checkpoint_smoke",
                input_mode="ab",
                hidden_dim=4,
                depth=1,
                lr=0.0,
                batch_size=2,
                epochs=1,
                warmup_epochs=0.0,
            )
            run_root = root / "runs"
            train_one(cfg, args, run_root, torch.device("cpu"), {"seed": 2026})
            run_dir = run_root / cfg.name
            self.assertTrue((run_dir / "head_best_angle.pt").is_file())
            self.assertTrue((run_dir / "head_best_final.pt").is_file())

    def test_cached_head_inference_writes_audited_output(self) -> None:
        try:
            import torch
            from pairuav.heads import RangeMLP, SixDofHead
            from pairuav.infer_cache import run_inference
        except Exception as exc:  # pragma: no cover - environment-dependent skip
            self.skipTest(f"PyTorch inference stack unavailable: {exc}")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            cache.mkdir()
            np.save(cache / "features.npy", np.zeros((3, 2, 4), dtype=np.float16))
            (cache / "meta.json").write_text(
                json.dumps({"samples": 3, "pooled_dim": 4, "dtype": "float16"}),
                encoding="utf-8",
            )

            angle_config = root / "angle.json"
            angle_config.write_text(
                json.dumps({
                    "name": "angle_test",
                    "input_mode": "ab",
                    "hidden_dim": 4,
                    "depth": 1,
                    "dropout": 0.0,
                }),
                encoding="utf-8",
            )
            range_config = root / "range.json"
            range_config.write_text(
                json.dumps({
                    "name": "range_test",
                    "input_mode": "ab",
                    "hidden_dim": 4,
                    "depth": 1,
                    "dropout": 0.0,
                }),
                encoding="utf-8",
            )
            angle_ckpt = root / "angle.pt"
            range_ckpt = root / "range.pt"
            torch.save(SixDofHead(feat_dim=4, hidden_dim=4, input_mode="ab", depth=1).state_dict(), angle_ckpt)
            range_model = RangeMLP(in_dim=8, hidden_dim=4, depth=1)
            range_model.init_mean(0.0)
            torch.save(range_model.state_dict(), range_ckpt)

            out = root / "result.txt"
            args = argparse.Namespace(
                feature_cache=cache,
                expected_rows=3,
                angle_ckpt=angle_ckpt,
                angle_config=angle_config,
                range_ckpt=range_ckpt,
                range_config=range_config,
                range2_ckpt=None,
                range2_config=None,
                gate_threshold=None,
                out=out,
                raw_heads_out=None,
                batch_size=2,
                device="cpu",
                matmul_precision="highest",
                seed=2026,
                deterministic=True,
                overwrite=False,
            )
            meta = run_inference(args)
            pred = np.loadtxt(out)
            self.assertEqual(pred.shape, (3, 2))
            np.testing.assert_allclose(pred, 0.0, atol=1e-6)
            self.assertEqual(meta["rows"], 3)
            self.assertTrue(out.with_suffix(".txt.meta.json").is_file())


if __name__ == "__main__":
    unittest.main()
