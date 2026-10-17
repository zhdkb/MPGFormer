import math
import unittest

import torch

from models.MPGFormer.modules.temporal import NonStationaryAttentionLayer


class TemporalAttentionTest(unittest.TestCase):
    def test_equation_seven_and_head_temperature_shape(self):
        layer = NonStationaryAttentionLayer(8, 2, 16, 0.0, 3)
        self.assertEqual(layer.factor[-1].out_features, 2)
        self.assertEqual(layer.factor[0].in_features, 4)
        q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[2.0, 1.0], [1.0, 2.0]]]])
        k = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[2.0, 0.0], [0.0, 2.0]]]])
        tau = torch.tensor([0.5, 2.0]).view(1, 2, 1, 1)
        expected = torch.softmax(tau * (q @ k.transpose(-2, -1)) / math.sqrt(2), dim=-1)
        torch.testing.assert_close(layer.attention_weights(q, k, tau), expected)
        result = layer(torch.randn(2, 5, 8), torch.randn(2, 5))
        self.assertEqual(result.shape, (2, 5, 8))

    def test_attention_statistics_use_raw_level_scale_slope_and_change(self):
        layer = NonStationaryAttentionLayer(8, 2, 16, 0.0, 3)
        history = torch.tensor([[1., 3., 5., 7., 9.], [10., 9., 8., 7., 6.]])
        expected = torch.tensor([
            [5., math.sqrt(8), 2., 2.],
            [8., math.sqrt(2), -1., 1.],
        ])
        torch.testing.assert_close(layer.raw_window_statistics(history), expected)

        seen = []
        hook = layer.factor[0].register_forward_pre_hook(
            lambda _module, args: seen.append(args[0].detach().clone()))
        try:
            layer(torch.randn(2, 5, 8), history)
        finally:
            hook.remove()
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0], expected)


if __name__ == "__main__":
    unittest.main()
