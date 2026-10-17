import unittest

import numpy as np

from protocol import mean_by_batch, pool_training_partition, report_test_metrics, split_workload
from models.MPGFormer.data import MPGFormerWindowDataset
from models.MPGFormer.modules.graph import DynamicGraphContrastiveEncoder


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

    def test_global_pool_uses_training_partition(self):
        ecw_train = split_workload(self.data, "ECW_08.npy")[0]
        np.testing.assert_array_equal(
            pool_training_partition(self.data, "ECW_08.npy"), ecw_train
        )
        np.testing.assert_array_equal(
            pool_training_partition(self.data, "ECW_08.npy", source_train_ratio=0.8),
            self.data[:, :80],
        )
        azure_train = split_workload(
            self.data, "tr_vm_official.npy", self.data + 100000
        )[0]
        np.testing.assert_array_equal(
            pool_training_partition(self.data, "tr_vm_official.npy"), azure_train
        )
        np.testing.assert_array_equal(
            pool_training_partition(self.data, "tr_materna_mt1.npy"), self.data
        )

    def test_partial_batch_does_not_get_full_batch_weight(self):
        self.assertAlmostEqual(mean_by_batch([1.0, 9.0], [3, 1]), 3.0)
        with self.assertRaises(ValueError):
            mean_by_batch([1.0, 2.0], [2])
        with self.assertRaises(ValueError):
            report_test_metrics("model", 0, "train.npy", "test.npy", np.nan, 0.1, 10.0)

    def test_future_measured_covariates_are_hidden(self):
        data = np.arange(72 * 6, dtype=np.float32).reshape(1, 72, 6)
        window = MPGFormerWindowDataset(data, input_size=6, static_start=6, stride=12)
        x, _, y, _ = window[0]
        np.testing.assert_array_equal(y.numpy(), data[0, 48:, 0])
        np.testing.assert_array_equal(x[:48].numpy(), data[0, :48])
        np.testing.assert_array_equal(x[48:, 1:4].numpy(), data[0, 48:, 1:4])
        self.assertTrue((x[48:, 0] == 0).all())
        self.assertTrue((x[48:, 4:] == 0).all())

    def test_graph_topk_remains_sparse_with_tied_similarities(self):
        import torch

        encoder = DynamicGraphContrastiveEncoder(8, 0, 0.0, topk=2)
        features = torch.ones(5, 48)
        adjacency = encoder._similarity_graph(features)
        self.assertTrue(torch.all((adjacency > 0).sum(dim=1) == 3))
        torch.testing.assert_close(adjacency.sum(dim=1), torch.ones(5))


if __name__ == "__main__":
    unittest.main()
