"""Run all single-component ablations of the current no-ERA5 model.

The completed direct-training result is the full-model reference. Every job
started by this script differs from that reference in exactly one component.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DATASETS = {
    "France": 16,
    "Australia": 16,
    "Zhejiang": 16,
    "Xinjiang": 1,
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FULL_MODEL = {
    "use_temporal_smooth": 1,
    "use_time_attn": 1,
    "use_var_attn": 1,
    "pointwise_loss": "huber",
    "missing_penalty": 2.0,
    "feature_weights": (1.0, 1.0, 1.5, 1.5, 1.0),
    "temporal_weight": 0.1,
    "wind_vector_weight": 0.025,
    "adaptive_temporal_scale": 1,
    "temporal_feature_weights": (1.0, 1.0, 0.5, 0.5, 1.0),
}

# Each entry changes exactly one full-model component. Values not listed here
# are inherited unchanged from FULL_MODEL.
ABLATIONS = (
    ("without_temporal_smooth", {"use_temporal_smooth": 0}),
    ("without_time_attention", {"use_time_attn": 0}),
    ("without_variable_attention", {"use_var_attn": 0}),
    ("mse_pointwise_reconstruction", {"pointwise_loss": "mse"}),
    ("without_missing_reweighting", {"missing_penalty": 0.0}),
    ("uniform_feature_weights", {"feature_weights": (1.0, 1.0, 1.0, 1.0, 1.0)}),
    ("without_adaptive_temporal_scale", {"adaptive_temporal_scale": 0}),
    ("without_temporal_innovation_loss", {"temporal_weight": 0.0}),
    ("without_wind_vector_loss", {"wind_vector_weight": 0.0}),
    ("uniform_temporal_feature_weights", {"temporal_feature_weights": (1.0, 1.0, 1.0, 1.0, 1.0)}),
)

COMMON_ARGS = {
    "lr": 5e-4,
    "hidden_dim": 256,
    "num_layers": 3,
    "num_heads": 4,
    "kernel_size": 3,
    "beta": 0.1,
    "train_missing_ratio": 0.2,
    "val_missing_ratio": 0.2,
    "train_missing_type": "point_mcar",
    "seq_len": 24,
    "patience": 20,
}


def write_manifest(path, manifest):
    with path.open("w") as handle:
        json.dump(manifest, handle, indent=2)


def make_command(dataset, batch_size, name, config, args, output_dir, summary_path):
    return [
        sys.executable, "-u", str(PROJECT_ROOT / "main.py"),
        "--dataset", dataset,
        "--batch_size", str(batch_size),
        "--lr", str(COMMON_ARGS["lr"]),
        "--hidden_dim", str(COMMON_ARGS["hidden_dim"]),
        "--num_layers", str(COMMON_ARGS["num_layers"]),
        "--num_heads", str(COMMON_ARGS["num_heads"]),
        "--kernel_size", str(COMMON_ARGS["kernel_size"]),
        "--use_era5", "0",
        "--use_temporal_smooth", str(config["use_temporal_smooth"]),
        "--use_time_attn", str(config["use_time_attn"]),
        "--use_var_attn", str(config["use_var_attn"]),
        "--beta", str(COMMON_ARGS["beta"]),
        "--pointwise_loss", config["pointwise_loss"],
        "--missing_penalty", str(config["missing_penalty"]),
        "--temporal_weight", str(config["temporal_weight"]),
        "--wind_vector_weight", str(config["wind_vector_weight"]),
        "--adaptive_temporal_scale", str(config["adaptive_temporal_scale"]),
        "--train_missing_ratio", str(COMMON_ARGS["train_missing_ratio"]),
        "--val_missing_ratio", str(COMMON_ARGS["val_missing_ratio"]),
        "--train_missing_type", COMMON_ARGS["train_missing_type"],
        "--seq_len", str(COMMON_ARGS["seq_len"]),
        "--patience", str(COMMON_ARGS["patience"]),
        "--epochs", str(args.epochs),
        "--num_runs", str(args.num_runs),
        "--selection_metric", "mae",
        "--feature_weights", *[str(value) for value in config["feature_weights"]],
        "--temporal_feature_weights", *[str(value) for value in config["temporal_feature_weights"]],
        "--ablation_name", name,
        "--output_dir", str(output_dir),
        "--summary_path", str(summary_path),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run every no-ERA5 single-component ablation across four regions."
    )
    parser.add_argument("--results-root", default="results/ablation_study")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--rerun-completed", action="store_true",
        help="Run jobs again even when their JSON summary already exists.",
    )
    args = parser.parse_args()
    if args.num_runs < 1 or args.epochs < 1:
        parser.error("--num-runs and --epochs must be positive.")

    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = {
        "description": "No-ERA5 single-component ablation study.",
        "full_model_reference": {
            "summary_root": "results/direct_training/final/adaptive_temporal_01_wind_0025",
            "config": FULL_MODEL,
        },
        "training": {"num_runs": args.num_runs, "epochs": args.epochs, "datasets": DATASETS},
        "ablations": {},
    }
    write_manifest(manifest_path, manifest)

    start_time = time.time()
    for name, override in ABLATIONS:
        config = {**FULL_MODEL, **override}
        manifest["ablations"][name] = {
            "changed_component": next(iter(override)),
            "override": override,
            "full_config": config,
            "datasets": {},
        }
        for dataset, batch_size in DATASETS.items():
            output_dir = root / name
            summary_path = output_dir / f"{dataset}.json"
            job = manifest["ablations"][name]["datasets"].setdefault(
                dataset, {"summary_path": str(summary_path)}
            )
            if summary_path.is_file() and not args.rerun_completed:
                job["status"] = "skipped_existing"
                write_manifest(manifest_path, manifest)
                print(f"Skipping completed job: {name} / {dataset}", flush=True)
                continue

            command = make_command(
                dataset, batch_size, name, config, args, output_dir, summary_path
            )
            job["status"] = "running"
            job["command"] = command
            write_manifest(manifest_path, manifest)
            print("\n" + "#" * 88, flush=True)
            print(
                f"Ablation={name} | Dataset={dataset} | batch_size={batch_size} | "
                f"runs={args.num_runs} | epochs={args.epochs}",
                flush=True,
            )
            print("Executing:", " ".join(command), flush=True)
            print("#" * 88, flush=True)
            try:
                subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            except subprocess.CalledProcessError as error:
                job["status"] = "failed"
                job["returncode"] = error.returncode
                write_manifest(manifest_path, manifest)
                raise
            job["status"] = "completed"
            write_manifest(manifest_path, manifest)

    manifest["completed"] = True
    manifest["elapsed_seconds"] = time.time() - start_time
    write_manifest(manifest_path, manifest)
    print(f"\nCompleted all ablations. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
