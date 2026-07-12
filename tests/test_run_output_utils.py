from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_output_utils import artifact_family_exists, resolve_experiment_output  # noqa: E402


class RunOutputUtilsTest(unittest.TestCase):
    def test_artifact_family_detects_sibling_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "ground_grid.json"
            self.assertFalse(artifact_family_exists(output))
            output.with_suffix(".path.npy").write_bytes(b"x")
            self.assertTrue(artifact_family_exists(output))

    def test_clean_explicit_output_is_honored(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new" / "result.json"
            resolved, run_dir, reason = resolve_experiment_output(output, Path(tmp), "topic")
            self.assertEqual(resolved, output.resolve())
            self.assertIsNone(run_dir)
            self.assertIsNone(reason)

    def test_existing_output_redirects_to_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requested = root / "old" / "ground_grid.json"
            requested.parent.mkdir()
            requested.write_text("old", encoding="utf-8")
            run_dir = root / "runs" / "unique"
            (run_dir / "assets").mkdir(parents=True)
            completed = SimpleNamespace(stdout=str(run_dir) + "\n")
            with patch("run_output_utils.subprocess.run", return_value=completed):
                output, allocated, reason = resolve_experiment_output(
                    requested, root, "topic", argv=["python", "smoke.py"]
                )
            self.assertEqual(output, run_dir / "assets" / "ground_grid.json")
            self.assertEqual(allocated, run_dir)
            self.assertIn("already exists", reason)
            self.assertEqual(requested.read_text(encoding="utf-8"), "old")
            self.assertTrue((run_dir / "cmd.txt").exists())


if __name__ == "__main__":
    unittest.main()
