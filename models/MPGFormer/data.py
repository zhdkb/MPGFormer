"""Dataset helpers for MPGFormer experiments."""

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MPGFormerWindowDataset(Dataset):
    """Create rolling forecasting windows from a 3-D workload array.

    Input arrays are expected to follow ``[num_series, num_periods, num_features]``.
    For every generated sample, future target and unavailable measured
    covariates are hidden. Calendar fields at columns 1 through 3 remain known.
    """

    def __init__(
        self,
        array,
        enc_len: int = 48,
        label_len: int = 12,
        pred_len: int = 24,
        input_size: Optional[int] = None,
        target_col: int = 0,
        static_start: Optional[int] = None,
        stride: int = 12,
    ):
        data = np.asarray(array, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError("array must have shape [num_series, num_periods, num_features]")

        self.enc_len = enc_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.target_col = target_col
        self.input_size = input_size or data.shape[-1]
        self.static_start = static_start
        self.samples = []
        self.static = []

        window = enc_len + pred_len
        for series_id in range(data.shape[0]):
            series_static = None
            if static_start is not None and static_start < data.shape[-1]:
                series_static = data[series_id, 0, static_start:]

            for end in range(window, data.shape[1] + 1, stride):
                block = data[series_id, end - window : end].copy()
                x = block[:, : self.input_size]
                y = block[-pred_len:, target_col].copy()

                # Only calendar features are known at forecast issuance.
                x[enc_len:, target_col] = 0.0
                if self.input_size > 4:
                    x[enc_len:, 4:] = 0.0
                self.samples.append((x, y, series_id))

                if series_static is not None:
                    self.static.append(series_static)

        self.has_static = static_start is not None and len(self.static) == len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, series_id = self.samples[idx]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        if self.has_static:
            static = torch.from_numpy(np.asarray(self.static[idx], dtype=np.float32)).float()
        else:
            static = torch.empty(0, dtype=torch.float32)

        return x, static, y, torch.tensor(series_id, dtype=torch.long)


def chronological_split(array, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Split each series along the time dimension without shuffling."""

    data = np.asarray(array, dtype=np.float32)
    train_len = int(data.shape[1] * train_ratio)
    return data[:, :train_len], data[:, train_len:]
