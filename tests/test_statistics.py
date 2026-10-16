import unittest
import io
import json
import tempfile
from contextlib import redirect_stdout

import numpy as np

from scripts.summarize_seed_runs import holm_adjust, read_records, summarize


class StatisticalSummaryTest(unittest.TestCase):
    def test_holm_adjustment_is_monotone_in_sorted_p_values(self):
        np.testing.assert_allclose(
            holm_adjust([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06]
        )

    def test_requested_model_filters_paired_comparisons(self):
        records = {}
        for seed in range(11):
            for name, mae in (("MPGFormer", 0.5), ("Informer", 0.6), ("PatchTST", 0.7)):
                records[("ECW", name, seed)] = {"mae": mae + seed * 0.001}
        output = io.StringIO()
        with redirect_stdout(output):
            summarize(records, baseline="Informer", model="MPGFormer")
        self.assertIn("MPGFormer vs Informer", output.getvalue())
        self.assertNotIn("PatchTST vs Informer", output.getvalue())

    def test_plain_jsonl_records_are_read(self):
        record = {"dataset": "MT1", "model": "PatchTST", "seed": 0,
                  "mse": 1.0, "mae": 0.5, "smape": 20.0}
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            self.assertEqual(read_records([stream.name]), {("MT1", "PatchTST", 0): record})


if __name__ == "__main__":
    unittest.main()
