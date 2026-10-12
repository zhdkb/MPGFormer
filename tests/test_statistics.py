import unittest

import numpy as np

from scripts.summarize_seed_runs import holm_adjust


class StatisticalSummaryTest(unittest.TestCase):
    def test_holm_adjustment_is_monotone_in_sorted_p_values(self):
        np.testing.assert_allclose(
            holm_adjust([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06]
        )


if __name__ == "__main__":
    unittest.main()
