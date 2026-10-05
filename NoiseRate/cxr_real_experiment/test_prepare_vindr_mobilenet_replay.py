import tempfile
import unittest
from pathlib import Path

import pandas as pd

from prepare_vindr_mobilenet_replay import prepare


class PrepareVinDrMobileNetReplayTests(unittest.TestCase):
    def test_requires_exact_sixty_run_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / ".prepare_complete").write_text("complete\n")
            (source / ".benchmark_verified").write_text("complete\n")
            pd.DataFrame([{"scenario_id": "clean", "seed": 13}]).to_csv(
                source / "scenario_manifest.csv", index=False
            )
            pd.DataFrame([{"scenario_id": "clean", "seed": 13}]).to_csv(
                source / "blind_run_manifest.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "60-run"):
                prepare(source, root / "output", root / "protocol", root / "engine")

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                prepare(root / "source", output, root / "protocol", root / "engine")


if __name__ == "__main__":
    unittest.main()
