import unittest

import numpy as np

from protocol import split_workload


class SplitProtocolTest(unittest.TestCase):
    def setUp(self):
        series = np.arange(10)[:, None, None]
        time = np.arange(100)[None, :, None]
        self.data = (series * 1000 + time).astype(np.float32)

    def test_ecw_is_chronological(self):
        train, val, test = split_workload(self.data, "ECW_08.npy")
        self.assertEqual((train.shape[1], val.shape[1], test.shape[1]), (64, 16, 20))
        np.testing.assert_array_equal(val, self.data[:, 64:80])
        np.testing.assert_array_equal(test, self.data[:, 80:])

    def test_new_trace_keeps_test_independent(self):
        held_out = self.data + 100000
        train, val, test = split_workload(self.data, "ECW_08.npy", held_out)
        np.testing.assert_array_equal(train, self.data[:, :80])
        np.testing.assert_array_equal(val, self.data[:, 80:])
        np.testing.assert_array_equal(test, held_out)

    def test_fixed_series_holdout(self):
        held_out = self.data + 100000
        train, val, test = split_workload(self.data, "tr_vm_official.npy", held_out)
        self.assertEqual((len(train), len(val)), (8, 2))
        self.assertFalse(set(train[:, 0, 0]) & set(val[:, 0, 0]))
        np.testing.assert_array_equal(test, held_out)
        np.testing.assert_array_equal(
            val, split_workload(self.data, "tr_vm_official.npy", held_out)[1]
        )

    def test_explicit_materna_validation(self):
        val_file = self.data + 100000
        test_file = self.data + 200000
        train, val, test = split_workload(self.data, "tr_materna_mt1.npy", test_file, val_file)
        np.testing.assert_array_equal(train, self.data)
        np.testing.assert_array_equal(val, val_file)
        np.testing.assert_array_equal(test, test_file)


if __name__ == "__main__":
    unittest.main()
