import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))
sys.path.insert(0, str(ROOT))

from smart_learning_app.live_service import (
    INFERENCE_STEP_SAMPLES,
    WINDOW_SAMPLES,
    ProductionInferenceWorker,
    ThinkGearLiveWorker,
)


class LiveServiceTest(unittest.TestCase):
    def test_first_raw_and_window_are_sample_index_driven(self):
        worker = ThinkGearLiveWorker(port=1)
        windows = []
        worker.window_ready.connect(windows.append)
        worker._consume_packet({
            "poorSignalLevel": 0,
            "eSense": {"attention": 55, "meditation": 48},
        })
        for index in range(WINDOW_SAMPLES + INFERENCE_STEP_SAMPLES):
            worker._consume_packet({"rawEeg": index % 100})
        # One result as soon as 30 s is ready, then one more after the 2 s step.
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[-1]["raw"].shape, (WINDOW_SAMPLES,))
        self.assertTrue(np.isfinite(windows[-1]["attention"]).all())
        self.assertEqual(windows[-1]["poor_signal"], 0)

    def test_auxiliary_fill_rejects_fully_missing_values(self):
        with self.assertRaisesRegex(ValueError, "Attention/Meditation"):
            ProductionInferenceWorker._fill(np.full(10, np.nan))

    def test_reconnect_clears_stale_window(self):
        worker = ThinkGearLiveWorker(port=1)
        worker._consume_packet({"rawEeg": 1})
        self.assertEqual(len(worker._raw), 1)
        worker._reset_stream()
        self.assertEqual(len(worker._raw), 0)
        self.assertIsNone(worker._first_raw_monotonic)


if __name__ == "__main__":
    unittest.main()
