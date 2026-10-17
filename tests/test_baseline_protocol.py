import unittest

import numpy as np

from models.baseline.workload_runner import Windows, SOURCES


class BaselineProtocolTest(unittest.TestCase):
    def test_complete_model_set_and_calendar_only_future_marks(self):
        self.assertEqual(set(SOURCES), {"PatchTST", "NS-Transformer", "Pathformer",
                                        "TimeMixer", "MSGNet", "CrossGNN", "TimesNet"})
        raw = np.arange(72 * 6, dtype=np.float32).reshape(1, 72, 6)
        windows = Windows(raw, raw, 48, 12, 24, 12)
        history, label, truth, _, marks = windows[0]
        self.assertEqual((len(history), len(label), len(truth)), (48, 12, 24))
        np.testing.assert_array_equal(marks[:, :3], raw[0, :, 1:4])
        self.assertTrue(np.all(marks[:, 3] == 0))
        self.assertNotIn(raw[0, 48, 0], marks[48])
        self.assertNotIn(raw[0, 48, 4], marks[48])


if __name__ == "__main__":
    unittest.main()
