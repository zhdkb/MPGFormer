"""Common workload protocol for the seven pinned upstream baseline implementations.

Run one model per process because several upstream projects use the same top-level
``models`` and ``layers`` package names.
"""

import argparse
import copy
import importlib
import json
import pickle
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from protocol import report_test_metrics, split_workload  # noqa: E402


SOURCES = {
    "PatchTST": ("PatchTST/PatchTST_supervised", "models.PatchTST"),
    "NS-Transformer": ("NS_Transformer", "ns_models.ns_Transformer"),
    "Pathformer": ("Pathformer", "models.PathFormer"),
    "TimeMixer": ("TimeMixer", "models.TimeMixer"),
    "MSGNet": ("MSGNet", "models.MSGNet"),
    "CrossGNN": ("CrossGNN", "models.CrossGNN"),
    "TimesNet": ("TimesNet", "models.TimesNet"),
}


def load_upstream_model(name, device):
    relative, module_name = SOURCES[name]
    source = Path(__file__).resolve().parent / relative
    if not source.is_dir() or not any(source.iterdir()):
        raise RuntimeError("Missing upstream source; run: git submodule update --init --recursive")
    sys.path.insert(0, str(source))
    # Two upstream implementations allocate parameters directly on CUDA during
    # construction. Apply only the device-string substitution in memory so their
    # pinned source can also run on a CPU-only host.
    if device.type == "cpu" and name in {"Pathformer", "CrossGNN"}:
        file = source / ("models/PathFormer.py" if name == "Pathformer" else "models/CrossGNN.py")
        original = file.read_text(encoding="utf-8")
        old = ("torch.device('cuda:{}'.format(configs.gpu))" if name == "Pathformer"
               else "'cuda:'+str(configs.gpu)")
        if original.count(old) != 1:
            raise RuntimeError(f"Upstream device expression changed in {file}")
        package_name = module_name.rpartition(".")[0]
        importlib.import_module(package_name)
        module = ModuleType(module_name)
        module.__file__ = str(file)
        module.__package__ = package_name
        sys.modules[module_name] = module
        exec(compile(original.replace(old, "torch.device('cpu')"), str(file), "exec"), module.__dict__)
        return module.Model
    return importlib.import_module(module_name).Model


def configuration(args, channels):
    """Pass the paper's window lengths and explicit architecture settings upstream."""
    return SimpleNamespace(
        task_name="long_term_forecast", seq_len=args.seq_len,
        label_len=args.label_len, pred_len=args.pred_len,
        enc_in=channels, dec_in=channels, c_out=channels, num_nodes=channels,
        d_model=args.d_model, d_ff=args.d_ff, n_heads=args.n_heads,
        e_layers=args.e_layers, d_layers=1, dropout=args.dropout,
        factor=1, activation="gelu", embed="timeF", freq="h",
        output_attention=False, p_hidden_dims=[128, 128], p_hidden_layers=2,
        patch_len=16, stride=8, padding_patch="end", revin=True,
        affine=False, subtract_last=False, decomposition=False,
        kernel_size=25, fc_dropout=args.dropout, head_dropout=args.dropout,
        individual=False, top_k=3, num_kernels=6,
        channel_independence=1, decomp_method="moving_avg", moving_avg=25,
        down_sampling_layers=2, down_sampling_window=2,
        down_sampling_method="avg", use_future_temporal_feature=0,
        use_norm=1, num_class=1,
        conv_channel=32, skip_channel=32, gcn_depth=2,
        propalpha=0.05, node_dim=10, subgraph_size=channels,
        layer_nums=3, k=2, num_experts_list=[4, 4, 4],
        patch_size_list=[[16, 12, 8, 6], [12, 8, 6, 4], [8, 6, 4, 2]],
        residual_connection=0, batch_norm=0, gpu=args.gpu,
        anti_ood=True, hidden=32, nvechidden=10, tvechidden=10,
        scale_number=3, tk=3, use_ngcn=True, use_tgcn=True,
    )


class Windows(Dataset):
    def __init__(self, scaled, raw, seq_len, label_len, pred_len, step):
        self.items = []
        for series in range(len(scaled)):
            for end in range(seq_len, scaled.shape[1] - pred_len + 1, step):
                history = scaled[series, end - seq_len:end].copy()
                label = scaled[series, end - label_len:end].copy()
                truth = scaled[series, end:end + pred_len, 0].copy()
                raw_truth = raw[series, end:end + pred_len, 0].copy()
                # Only calendar columns 1:4 are exposed from the forecast horizon.
                marks = np.zeros((seq_len + pred_len, 4), dtype=np.float32)
                n_calendar = min(3, scaled.shape[-1] - 1)
                if n_calendar:
                    marks[:, :n_calendar] = scaled[series, end - seq_len:end + pred_len, 1:1 + n_calendar]
                self.items.append((history, label, truth, raw_truth, marks))
        if not self.items:
            raise ValueError("No windows; each partition needs at least seq_len + pred_len points")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def load_array(path):
    if str(path).endswith(".pkl"):
        with open(path, "rb") as stream:
            array = pickle.load(stream)
    else:
        array = np.load(path, allow_pickle=True)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite [series,time,features] array: {path}")
    return array


def prepare(args):
    train, val, test = split_workload(
        load_array(args.train_data), args.train_data,
        load_array(args.test_data) if args.test_data else None,
        load_array(args.val_data) if args.val_data else None,
    )
    if args.input_size > train.shape[-1]:
        raise ValueError("input_size exceeds the dataset feature count")
    # Match the existing workload baseline entries' Azure unit convention.
    names = " ".join(Path(p).name.lower() for p in (args.train_data, args.test_data) if p)
    if "azure" in names or "_vm" in names:
        for partition in (train, val, test):
            partition[:, :, 0] /= 10.0
    partitions = [part[:, :, :args.input_size].copy() for part in (train, val, test)]
    scaler = MinMaxScaler().fit(partitions[0].reshape(-1, args.input_size))
    loaders = []
    for index, part in enumerate(partitions):
        scaled = scaler.transform(part.reshape(-1, args.input_size)).reshape(part.shape).astype(np.float32)
        windows = Windows(scaled, part, args.seq_len, args.label_len, args.pred_len,
                          args.step if index == 0 else args.test_step)
        loaders.append(DataLoader(windows, batch_size=args.batch_size, shuffle=index == 0))
    return loaders, scaler


def forecast(name, model, history, label, marks, label_len, pred_len, return_aux=False):
    auxiliary = None
    if name in {"PatchTST", "Pathformer", "CrossGNN"}:
        output = model(history)
        if isinstance(output, tuple):
            if name == "Pathformer":
                auxiliary = output[1]
            output = output[0]
    else:
        decoder = torch.cat([label, label.new_zeros(label.shape[0], pred_len, label.shape[-1])], dim=1)
        enc_mark = marks[:, :history.shape[1]]
        dec_mark = marks[:, -label_len - pred_len:]
        output = model(history, enc_mark, decoder, dec_mark)
        if isinstance(output, tuple):
            output = output[0]
    if output.shape[:2] != (history.shape[0], pred_len) or output.shape[-1] < 1:
        raise RuntimeError(f"Unexpected {name} output shape: {tuple(output.shape)}")
    prediction = output[:, :, 0]
    return (prediction, auxiliary) if return_aux else prediction


def evaluate(name, model, loader, device, scaler, args):
    model.eval()
    squared = absolute = percentage = count = 0.0
    target_min, target_scale = float(scaler.min_[0]), float(scaler.scale_[0])
    with torch.no_grad():
        for history, label, _, raw_truth, marks in loader:
            prediction = forecast(name, model, history.to(device), label.to(device),
                                  marks.to(device), args.label_len, args.pred_len)
            pred_raw = (prediction.cpu().numpy() - target_min) / target_scale
            truth = raw_truth.numpy()
            delta = pred_raw - truth
            squared += float(np.square(delta).sum())
            absolute += float(np.abs(delta).sum())
            percentage += float((200 * np.abs(delta) / np.maximum(np.abs(pred_raw) + np.abs(truth), 1e-8)).sum())
            count += delta.size
    return squared / count, absolute / count, percentage / count


def run(args):
    if args.seq_len != 48 or args.label_len != 12 or args.pred_len != 24:
        print("Note: window lengths differ from the paper's 48/12/24 protocol")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.cpu else "cpu")
    loaders, scaler = prepare(args)
    cls = load_upstream_model(args.model, device)
    model = cls(configuration(args, args.input_size)).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_mse, best_state, best_epoch = float("inf"), None, None
    dataset_id = Path(args.test_data).stem if args.test_data else Path(args.train_data).stem
    output_dir = Path(args.save_dir) / args.model / dataset_id / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        for history, label, truth, _, marks in loaders[0]:
            history, label, truth, marks = (item.to(device) for item in (history, label, truth, marks))
            pred, auxiliary = forecast(args.model, model, history, label, marks,
                                       args.label_len, args.pred_len, return_aux=True)
            loss = torch.nn.functional.mse_loss(pred, truth)
            if auxiliary is not None:
                loss = loss + auxiliary
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_mse, _, _ = evaluate(args.model, model, loaders[1], device, scaler, args)
        print(f"epoch={epoch + 1} validation_mse={val_mse:.8f}", flush=True)
        if val_mse < best_mse:
            best_mse, best_epoch = val_mse, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_dir / "best_state.pt")
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was selected")
    model.load_state_dict(best_state)
    mse, mae, smape = evaluate(args.model, model, loaders[2], device, scaler, args)
    (output_dir / "metrics.json").write_text(json.dumps({"seed": args.seed, "best_epoch": best_epoch,
        "validation_mse": best_mse, "test_mse": mse, "test_mae": mae, "test_smape": smape}, indent=2) + "\n")
    report_test_metrics(args.model, args.seed, args.train_data, args.test_data, mse, mae, smape)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SOURCES, required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--test_data", default="")
    parser.add_argument("--val_data", default="")
    parser.add_argument("--input_size", type=int, required=True)
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--label_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--step", type=int, default=12)
    parser.add_argument("--test_step", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save_dir", default="checkpoints/official_baselines")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
