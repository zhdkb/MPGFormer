"""Run workload models with matched seeds 0 through 10."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRIES = {
    "MPGFormer": ("models/MPGFormer", "train_mpgformer.py", "epochs", False),
    "Autoformer": ("models/baseline/auto_informer", "auto_info_workload_main.py", "train_epochs", False),
    "Informer": ("models/baseline/Informer", "informer_workload_main.py", "train_epochs", False),
    "Fremer": ("models/baseline/Fremer", "fremer_workload_main.py", "epochs", True),
    "iTransformer": ("models/baseline/iTransformer", "itransformer_workload_main.py", "train_epochs", True),
    "PDF": ("models/baseline/PDF", "pdf_workload_main.py", "epochs", True),
    "TFEGRU": ("models/baseline/TFEGRU", "tfegru_workload_main.py", "epochs", True),
    "SOFTS": ("models/baseline/SOFTS", "softs_workload_main.py", "epochs", True),
    "MICN": ("models/baseline/MICN", "micn_workload_main.py", "epochs", True),
    "PatchTST": ("models/baseline/PatchTST", "patchtst_workload_main.py", "epochs", True),
    "NS-Transformer": ("models/baseline/NS_Transformer", "ns_transformer_workload_main.py", "epochs", True),
    "Pathformer": ("models/baseline/Pathformer", "pathformer_workload_main.py", "epochs", True),
    "TimeMixer": ("models/baseline/TimeMixer", "timemixer_workload_main.py", "epochs", True),
    "MSGNet": ("models/baseline/MSGNet", "msgnet_workload_main.py", "epochs", True),
    "CrossGNN": ("models/baseline/CrossGNN", "crossgnn_workload_main.py", "epochs", True),
    "TimesNet": ("models/baseline/TimesNet", "timesnet_workload_main.py", "epochs", True),
}
DATASETS = {
    "ECW": ("ECW_08.npy", None, None, 4, 300, 64, 12),
    "NewInfras": ("ECW_08.npy", None, "ECW_newmac.pkl", 4, 300, 32, 2),
    "NewApp": ("ECW_08.npy", None, "ECW_newapp.pkl", 4, 300, 64, 2),
    "Azure": ("tr_vm_official.npy", None, "te_vm_official.npy", 4, 300, 64, 12),
    "GoogleV1": ("tr_google_v1.npy", None, "te_google_v1.npy", 6, 300, 64, 12),
    "MT1": ("tr_materna_mt1.npy", "va_materna_mt1.npy", "te_materna_mt1.npy", 10, 30, 64, 12),
    "MT2": ("tr_materna_mt2.npy", "va_materna_mt2.npy", "te_materna_mt2.npy", 10, 30, 64, 12),
    "MT3": ("tr_materna_mt3.npy", "va_materna_mt3.npy", "te_materna_mt3.npy", 10, 30, 64, 12),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--models", nargs="+", choices=ENTRIES, default=list(ENTRIES))
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data")
    parser.add_argument("--log_dir", type=Path, default=ROOT / "logs/seed_runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(11)))
    parser.add_argument("--epochs", type=int, help="Override only for debugging")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be distinct nonnegative integers")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be positive")
    train, val, test, channels, epochs, batch, test_step = DATASETS[args.dataset]
    required = [args.data_dir / name for name in (train, val, test) if name]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing processed data: " + ", ".join(missing))
    output = args.log_dir / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for model in args.models:
        workdir_name, entry, epoch_flag, input_size_flag = ENTRIES[model]
        workdir = ROOT / workdir_name
        for seed in args.seeds:
            command = [sys.executable, entry,
                       "--data" if model == "MPGFormer" else "--train_data", str(args.data_dir / train),
                       f"--{epoch_flag}", str(args.epochs or epochs),
                       "--batch_size", str(batch), "--test_step", str(test_step),
                       "--seed", str(seed), "--save_dir", str(output / "checkpoints" / model / f"seed_{seed}")]
            if input_size_flag or model == "MPGFormer":
                command += ["--input_size", str(channels)]
            if model == "MPGFormer":
                command += ["--static_start", str(channels), "--enc_len", "48",
                            "--label_len", "12", "--pred_len", "24"]
            if val:
                command += ["--val_data", str(args.data_dir / val)]
            if test:
                command += ["--test_data", str(args.data_dir / test)]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
            log_path = output / f"{model}_seed_{seed}.log"
            with log_path.open("w") as stream:
                process = subprocess.Popen(command, cwd=workdir, env=environment,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                metrics = []
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    if line.startswith("TEST_METRICS "):
                        metrics.append(json.loads(line.removeprefix("TEST_METRICS ")))
                returncode = process.wait()
            if returncode or len(metrics) != 1:
                raise RuntimeError(f"{model} seed {seed} failed; inspect {log_path}")
            expected_dataset = Path(test).stem if test else "ECW"
            if (metrics[0].get("model") != model or metrics[0].get("seed") != seed
                    or metrics[0].get("dataset") != expected_dataset):
                raise RuntimeError(f"{model} seed {seed} reported inconsistent metrics; inspect {log_path}")
            records.extend(metrics)
            (output / "runs.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records))
    print(f"Wrote {len(records)} test records to {output / 'runs.jsonl'}")


if __name__ == "__main__":
    main()
