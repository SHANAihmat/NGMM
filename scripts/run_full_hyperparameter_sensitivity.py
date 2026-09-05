#!/usr/bin/env python3
"""Run dense, one-factor hyperparameter sensitivity experiments.

For every region, the baseline is the completed robust-search configuration.
Each trial changes exactly one scalar hyperparameter; all remaining values are
copied unchanged from that region's best configuration.  Validation-only
evaluation is used so the test split remains untouched until final reporting.
The manifest is updated after every subprocess, making the sweep resumable.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("France", "Australia", "Zhejiang", "Xinjiang")
DATASET_BATCH = {"France": 16, "Australia": 16, "Zhejiang": 16, "Xinjiang": 1}
CORE_PARAMETERS = (
    "hidden_dim", "num_layers", "num_heads", "kernel_size", "seq_len",
    "lr", "weight_decay", "temporal_weight",
)

# Every list contains at least four values. The default/selected value is
# included automatically when it is not already present for a region.
SWEEP_VALUES = {
    "hidden_dim": [64, 128, 256, 512],
    "num_layers": [1, 2, 3, 4],
    "num_heads": [1, 2, 4, 8],
    "kernel_size": [3, 5, 7, 9],
    "seq_len": [12, 24, 48, 72],
    "lr": [1e-4, 3e-4, 5e-4, 8e-4],
    "weight_decay": [1e-5, 1e-4, 3e-4, 1e-3],
    "beta": [0.05, 0.1, 0.2, 0.4],
    "missing_penalty": [0.0, 1.0, 2.0, 3.0],
    "temporal_weight": [0.025, 0.05, 0.1, 0.2],
    "wind_vector_weight": [0.005, 0.01, 0.025, 0.05],
}


def load_best(dataset):
    path = ROOT / "results" / "robust_hyperparameter_search" / dataset / "best_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing robust-search configuration: {path}")
    payload = json.loads(path.read_text())
    return payload["config"], path


def equal(a, b):
    return a == b


def values_for(parameter, baseline):
    values = list(SWEEP_VALUES[parameter])
    selected = baseline[parameter]
    if not any(equal(selected, x) for x in values):
        values.append(selected)
    # Preserve numeric order in plots and remove duplicates.
    return sorted(set(values))


def scalar_args(config):
    """Convert the model/training config to main.py command-line arguments."""
    keys = (
        "use_era5", "use_temporal_smooth", "use_time_attn", "use_var_attn",
        "hidden_dim", "num_layers", "num_heads", "kernel_size", "seq_len",
        "train_stride", "lr", "weight_decay", "grad_clip_norm",
        "scheduler_factor", "scheduler_patience", "min_lr", "train_missing_ratio",
        "train_missing_type", "use_augmentation", "aug_noise_std", "aug_scale_jitter",
        "aug_max_shift", "aug_dropout_rate", "aug_prob", "pointwise_loss", "beta",
        "missing_penalty", "temporal_weight", "wind_vector_weight",
        "adaptive_temporal_scale", "temporal_scale_floor",
    )
    args = []
    for key in keys:
        if key in config:
            args.extend([f"--{key}", str(config[key])])
    args.extend(["--feature_weights", *[str(x) for x in config.get("feature_weights", [1.0] * 5)]])
    args.extend(["--temporal_feature_weights", *[str(x) for x in config.get("temporal_feature_weights", [1.0, 1.0, 0.5, 0.5, 1.0])]])
    return args


def command(dataset, config, parameter, value, output_dir, summary_path, args):
    trial = dict(config)
    trial[parameter] = value
    if parameter == "seq_len":
        trial["train_stride"] = int(value) // 2
    trial["batch_size"] = args.batch_size or DATASET_BATCH[dataset]
    cmd = [sys.executable, "-u", str(ROOT / "main.py"), "--dataset", dataset]
    cmd += scalar_args(trial)
    cmd += ["--batch_size", str(trial["batch_size"]), "--epochs", str(args.epochs),
            "--num_runs", str(args.num_runs), "--patience", str(args.patience),
            "--selection_metric", "mae", "--validation_missing_types", *args.validation_types,
            "--validation_missing_ratios", *[str(x) for x in args.validation_ratios],
            "--validation_worst_weight", str(args.validation_worst_weight),
            "--selection_only", "--ablation_name", f"sensitivity_{parameter}_{value:g}",
            "--output_dir", str(output_dir), "--summary_path", str(summary_path)]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/hyperparameter_sensitivity_full")
    parser.add_argument("--run-id", default=None, help="Stable output folder name, useful for nohup jobs.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=("France", "Zhejiang"),
                        help="MN/France and SM/Zhejiang by default.")
    parser.add_argument("--parameters", nargs="+", choices=tuple(SWEEP_VALUES), default=CORE_PARAMETERS,
                        help="Eight paper-critical hyperparameters by default.")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Runs per value for sensitivity screening; use 5 only for final confirmation.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--validation-types", nargs="+", default=["point_mcar", "point_mar", "seq_mcar", "block_2d"])
    parser.add_argument("--validation-ratios", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    parser.add_argument("--validation-worst-weight", type=float, default=0.5)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop immediately after a failed trial. The default records the failure and continues.")
    parser.add_argument("--dry-run", action="store_true", help="Write the manifest and commands without launching training.")
    args = parser.parse_args()
    if args.num_runs < 1 or args.epochs < 1:
        parser.error("--num-runs and --epochs must be positive")
    if len(args.validation_types) != 4 or any(not 0 < x < 1 for x in args.validation_ratios):
        parser.error("Use four validation types and ratios in (0,1)")

    stamp = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = ROOT / args.results_root / stamp
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not args.rerun_completed:
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"protocol": {"selection": "validation only", "num_runs": args.num_runs,
            "epochs": args.epochs, "validation_types": args.validation_types,
            "validation_ratios": args.validation_ratios, "one_factor": True,
            "source_configs": "results/robust_hyperparameter_search/<dataset>/best_config.json"},
            "sweeps": {}}
        manifest_path.write_text(json.dumps(manifest, indent=2))

    for dataset in args.datasets:
        baseline, source = load_best(dataset)
        for parameter in args.parameters:
            values = values_for(parameter, baseline)
            sweep_key = f"{dataset}/{parameter}"
            sweep = manifest["sweeps"].setdefault(sweep_key, {"trials": {}})
            sweep.update({"baseline": baseline[parameter], "values": values, "source": str(source)})
            for value in values:
                tag = f"{dataset}_{parameter}_{value:g}"
                trial_dir = root / dataset / parameter
                trial_dir.mkdir(parents=True, exist_ok=True)
                summary = trial_dir / f"value_{value:g}.json"
                log = trial_dir / f"value_{value:g}.log"
                record = sweep["trials"].setdefault(str(value), {"summary": str(summary), "log": str(log), "value": value})
                if summary.exists() and not args.rerun_completed:
                    record["status"] = "skipped_existing"; manifest_path.write_text(json.dumps(manifest, indent=2)); continue
                cmd = command(dataset, baseline, parameter, value, trial_dir, summary, args)
                record.update({"status": "running", "command": cmd})
                manifest_path.write_text(json.dumps(manifest, indent=2))
                if args.dry_run:
                    record["status"] = "dry_run"
                    manifest_path.write_text(json.dumps(manifest, indent=2))
                    continue
                print(f"Running {tag}: {' '.join(cmd)}", flush=True)
                with log.open("w") as handle:
                    result = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                record["status"] = "completed" if result.returncode == 0 else "failed"
                record["returncode"] = result.returncode
                if result.returncode != 0:
                    tail = log.read_text(errors="replace").splitlines()[-30:]
                    record["error_tail"] = "\n".join(tail)
                manifest_path.write_text(json.dumps(manifest, indent=2))
                if result.returncode != 0 and args.stop_on_error:
                    raise RuntimeError(f"Sensitivity trial failed: {tag}; see {log}")
                if result.returncode != 0:
                    print(f"FAILED {tag}; continuing. See {log}", flush=True)
    manifest["completed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Completed full sensitivity sweep. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
