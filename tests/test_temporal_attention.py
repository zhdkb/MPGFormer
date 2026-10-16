import math
import unittest

import torch

from models.MPGFormer.modules.temporal import NonStationaryAttentionLayer


class TemporalAttentionTest(unittest.TestCase):
    def test_equation_seven_and_head_temperature_shape(self):
        layer = NonStationaryAttentionLayer(8, 2, 16, 0.0, 3)
        self.assertEqual(layer.factor[-1].out_features, 2)
        q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[2.0, 1.0], [1.0, 2.0]]]])
        k = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[2.0, 0.0], [0.0, 2.0]]]])
        tau = torch.tensor([0.5, 2.0]).view(1, 2, 1, 1)
        expected = torch.softmax(tau * (q @ k.transpose(-2, -1)) / math.sqrt(2), dim=-1)
        torch.testing.assert_close(layer.attention_weights(q, k, tau), expected)
        result = layer(torch.randn(2, 5, 8), torch.randn(2, 5))
        self.assertEqual(result.shape, (2, 5, 8))


if __name__ == "__main__":
    unittest.main()
