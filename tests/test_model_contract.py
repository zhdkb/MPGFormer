import unittest

import torch

from models.MPGFormer.mpgformer import MPGFormer
from models.MPGFormer.losses import MPGFormerLoss


class MPGFormerContractTest(unittest.TestCase):
    def test_future_targets_are_masked_and_outputs_follow_paper_shapes(self):
        torch.manual_seed(7)
        model = MPGFormer(
            input_size=4, enc_seq_len=48, label_len=12, pred_len=24,
            d_model=16, d_ff=32, n_heads=4, e_layers=1, d_layers=1,
            dropout=0.0, num_prototypes=4, graph_topk=1,
        ).eval()
        x = torch.randn(2, 72, 4)
        changed = x.clone()
        changed[:, 48:, 0] += 1000.0

        with torch.no_grad():
            output = model(x)
            masked_output = model(changed)

        torch.testing.assert_close(output.point, masked_output.point)
        torch.testing.assert_close(output.quantiles, masked_output.quantiles)
        self.assertEqual(output.point.shape, (2, 24))
        self.assertEqual(output.quantiles.shape, (2, 24, 3))
        self.assertTrue(torch.all(output.quantiles[..., 0] <= output.quantiles[..., 1]))
        self.assertTrue(torch.all(output.quantiles[..., 1] <= output.quantiles[..., 2]))
        torch.testing.assert_close(output.router_weights.sum(-1), torch.ones(2))

    def test_composite_loss_backpropagates(self):
        torch.manual_seed(11)
        model = MPGFormer(
            input_size=4, d_model=16, d_ff=32, n_heads=4,
            e_layers=1, d_layers=1, dropout=0.0, num_prototypes=4,
        )
        output = model(torch.randn(2, 72, 4))
        loss, _ = MPGFormerLoss()(output, torch.randn(2, 24))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.long_layers[0].qkv.weight.grad).all())
        self.assertTrue(torch.isfinite(model.prototype_bank.prototype.grad).all())


if __name__ == "__main__":
    unittest.main()
