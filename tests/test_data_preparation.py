import random
import unittest

import numpy as np

from scripts.prepare_azure_vm_dataset import split_segments_by_source


class AzureSourceSplitTest(unittest.TestCase):
    def test_multiple_segments_of_one_source_stay_together(self):
        segments = [
            (source, np.full((72, 4), source, dtype=np.float32))
            for source in range(4)
            for _ in range(3)
        ]
        train, test = split_segments_by_source(
            segments, train_ratio=0.75, max_segments=0, rng=random.Random(2)
        )
        train_sources = set(train[:, 0, 0])
        test_sources = set(test[:, 0, 0])
        self.assertFalse(train_sources & test_sources)
        self.assertEqual(train_sources | test_sources, set(range(4)))


if __name__ == "__main__":
    unittest.main()
