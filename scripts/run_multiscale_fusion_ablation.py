"""Ablate multi-scale temporal fusion across all meteorological regions.

The three variants differ only in information flow inside
``MultiScaleTemporalBlock``:

* independent: parallel multi-scale features without coarse-to-fine guidance;
* cascade: mandatory coarse-to-fine guidance without a local fallback;
* gated_cascade: proposed adaptive mixture of both representations.

Every run uses the robust-validation protocol and a fixed region-specific
training configuration selected before this architectural ablation.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DATASETS = ("France", "Australia", "Zhejiang", "Xinjiang")
VALIDATION_TYPES = ("point_mcar", "point_mar", "seq_mcar", "block_2d")
FUSION_VARIANTS = (
    ("independent", "parallel independent multi-scale baseline"),
    ("cascade", "coarse-to-fine cascade without adaptive gating"),
    ("gated_cascade", "proposed gated coarse-to-fine cascade"),
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMMON_CONFIG = {
    "use_era5": 0,
    "use_temporal_smooth": 1,
    "use_time_attn": 1,
    "use_var_attn": 1,
    "hidden_dim": 256,
    "num_layers": 3,
    "num_heads": 4,
    "kernel_size": 3,
    "seq_len": 24,
    "train_stride": 12,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "grad_clip_norm": 1.0,
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    "min_lr": 0.0,
    "train_missing_ratio": 0.2,
    "train_missing_type": "point_mcar",
    "use_augmentation": 0,
    "aug_noise_std": 0.01,
    "aug_scale_jitter": 0.05,
    "aug_max_shift": 1,
    "aug_dropout_rate": 0.1,
    "aug_prob": 0.5,
    "pointwise_loss": "huber",
    "beta": 0.1,
    "missing_penalty": 2.0,
    "temporal_weight": 0.1,
    "wind_vector_weight": 0.025,
    "adaptive_temporal_scale": 1,
    "temporal_scale_floor": 0.05,
    "feature_weights": [1.0, 1.0, 1.0, 1.0, 1.0],
    "temporal_feature_weights": [1.0, 1.0, 0.5, 0.5, 1.0],
}

# These are the region-specific robust-HPO selections. The fusion mode below
# is the sole experimental change from each region's selected configuration.
REGION_OVERRIDES = {
    "France": {"batch_size": 16, "kernel_size": 5},
    "Australia": {"batch_size": 16, "lr": 8e-4},
    "Zhejiang": {"batch_size": 16, "lr": 8e-4},
    "Xinjiang": {"batch_size": 1, "weight_decay": 3e-4},
}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def region_config(dataset):
    return {**COMMON_CONFIG, **REGION_OVERRIDES[dataset]}


def make_command(dataset, mode, args, output_dir, summary_path):
    config = region_config(dataset)
    command = [
        sys.executable, "-u", str(PROJECT_ROOT / "main.py"),
        "--dataset", dataset,
        "--epochs", str(args.epochs),
        "--num_runs", str(args.num_runs),
        "--patience", str(args.patience),
        "--selection_metric", "mae",
        "--validation_missing_types", *args.validation_types,
        "--validation_missing_ratios", *map(str, args.validation_ratios),
        "--validation_worst_weight", str(args.worst_condition_weight),
        "--temporal_fusion_mode", mode,
        "--ablation_name", "multiscale_" + mode,
        "--output_dir", str(output_dir),
        "--summary_path", str(summary_path),
    ]
    scalar_keys = (
        "use_era5", "use_temporal_smooth", "use_time_attn", "use_var_attn",
        "hidden_dim", "num_layers", "num_heads", "kernel_size", "seq_len",
        "train_stride", "batch_size", "lr", "weight_decay", "grad_clip_norm",
        "scheduler_factor", "scheduler_patience", "min_lr", "train_missing_ratio",
        "train_missing_type", "use_augmentation", "aug_noise_std", "aug_scale_jitter",
        "aug_max_shift", "aug_dropout_rate", "aug_prob", "pointwise_loss", "beta",
        "missing_penalty", "temporal_weight", "wind_vector_weight",
        "adaptive_temporal_scale", "temporal_scale_floor",
    )
    for key in scalar_keys:
        command.extend(["--" + key, str(config[key])])
    command.extend(["--feature_weights", *map(str, config["feature_weights"])])
    command.extend(["--temporal_feature_weights", *map(str, config["temporal_feature_weights"])])
    return command


def summary_row(path):
    summary = json.loads(path.read_text())
    test_mae = [condition["mae"]["mean"] for condition in summary["test"].values()]
    return {
        "parameters": summary["model_config"].get("num_params"),
        "validation_selection_score": summary["validation"]["selection_score"],
        "test_average_mae": sum(test_mae) / len(test_mae),
        "test_worst_mae": max(test_mae),
        "test_conditions": len(test_mae),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run independent, cascade, and gated-cascade multi-scale ablations for all regions."
    )
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--results-root", default="results/multiscale_fusion_ablation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--validation-types", nargs="+", choices=VALIDATION_TYPES, default=list(VALIDATION_TYPES))
    parser.add_argument("--validation-ratios", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    parser.add_argument("--worst-condition-weight", type=float, default=0.5)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.num_runs < 1 or args.patience < 1:
        parser.error("--epochs, --num-runs, and --patience must be positive.")
    if not args.validation_ratios or any(not 0 < ratio < 1 for ratio in args.validation_ratios):
        parser.error("--validation-ratios must be in (0, 1).")
    if args.worst_condition_weight < 0:
        parser.error("--worst-condition-weight must be non-negative.")

    root = Path(args.results_root)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not args.rerun_completed:
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "description": "Controlled ablation of multi-scale temporal fusion.",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "training": {
                "epochs": args.epochs,
                "num_runs": args.num_runs,
                "patience": args.patience,
                "validation_types": args.validation_types,
                "validation_ratios": args.validation_ratios,
                "worst_condition_weight": args.worst_condition_weight,
            },
            "variants": {mode: description for mode, description in FUSION_VARIANTS},
            "region_configs": {dataset: region_config(dataset) for dataset in DATASETS},
            "jobs": {},
        }
        write_json(manifest_path, manifest)

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    started = time.time()
    outcomes = {}
    for dataset in datasets:
        for mode, description in FUSION_VARIANTS:
            job_key = dataset + "/" + mode
            output_dir = root / dataset / mode
            summary_path = output_dir / "summary.json"
            job = manifest["jobs"].setdefault(job_key, {"summary_path": str(summary_path)})
            if summary_path.is_file() and not args.rerun_completed:
                job["status"] = "completed"
                job["summary"] = summary_row(summary_path)
                write_json(manifest_path, manifest)
                print(f"Skipping completed job: {job_key}", flush=True)
                continue

            command = make_command(dataset, mode, args, output_dir, summary_path)
            job.update({"status": "running", "description": description, "command": command})
            write_json(manifest_path, manifest)
            print("\n" + "#" * 88, flush=True)
            print(f"Dataset={dataset} | mode={mode} | {description}", flush=True)
            print("Executing:", " ".join(command), flush=True)
            print("#" * 88, flush=True)
            if args.dry_run:
                job["status"] = "dry_run"
                write_json(manifest_path, manifest)
                continue
            try:
                subprocess.run(command, check=True, cwd=PROJECT_ROOT)
                job.update({"status": "completed", "summary": summary_row(summary_path)})
            except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as error:
                job.update({"status": "failed", "error": str(error)})
                outcomes[job_key] = "failed"
                write_json(manifest_path, manifest)
                if args.stop_on_error:
                    raise
            else:
                outcomes[job_key] = "completed"
                write_json(manifest_path, manifest)

    comparison = {}
    for dataset in DATASETS:
        rows = []
        for mode, _ in FUSION_VARIANTS:
            job = manifest["jobs"].get(dataset + "/" + mode, {})
            if job.get("status") == "completed":
                rows.append({"mode": mode, **job["summary"]})
        if rows:
            comparison[dataset] = rows
    write_json(root / "comparison.json", comparison)
    manifest["elapsed_seconds"] = time.time() - started
    manifest["completed"] = all(
        manifest["jobs"].get(dataset + "/" + mode, {}).get("status") == "completed"
        for dataset in datasets for mode, _ in FUSION_VARIANTS
    )
    write_json(manifest_path, manifest)
    print(f"\nComparison summary: {root / 'comparison.json'}", flush=True)
    if any(status == "failed" for status in outcomes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
