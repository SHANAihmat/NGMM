"""Robust, all-region hyperparameter search for the meteorological model.

Unlike the earlier point-MCAR-only search, this script selects checkpoints on
a validation suite containing point, sequential, and block missingness. It
starts from the experimentally supported full model and searches a focused
neighborhood, so all paper-critical modules remain enabled.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DATASETS = ("France", "Australia", "Zhejiang", "Xinjiang")
VALIDATION_TYPES = ("point_mcar", "point_mar", "seq_mcar", "block_2d")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_CONFIG = {
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


def with_overrides(name, **overrides):
    config = {**BASE_CONFIG, **overrides}
    return {"name": name, "config": config}


# Every candidate retains the full temporal, variable, and multi-scale model.
# The baseline is included explicitly so random sampling cannot omit it.
CANDIDATES = (
    with_overrides("full_model_baseline"),
    with_overrides("kernel_5", kernel_size=5),
    with_overrides("long_context_48", seq_len=48, train_stride=24, kernel_size=5),
    with_overrides("long_context_72", seq_len=72, train_stride=36, kernel_size=7),
    with_overrides("layers_4", num_layers=4),
    with_overrides("hidden_192", hidden_dim=192),
    with_overrides("lr_3e4", lr=3e-4),
    with_overrides("lr_8e4", lr=8e-4),
    with_overrides("weight_decay_1e5", weight_decay=1e-5),
    with_overrides("weight_decay_3e4", weight_decay=3e-4),
    with_overrides("temporal_weight_005", temporal_weight=0.05),
    with_overrides("temporal_weight_02", temporal_weight=0.2),
    with_overrides("wind_weight_001", wind_vector_weight=0.01),
    with_overrides("wind_weight_005", wind_vector_weight=0.05),
    with_overrides("missing_penalty_3", missing_penalty=3.0),
    with_overrides("wind_feature_weight_15", feature_weights=[1.0, 1.0, 1.5, 1.5, 1.0]),
)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def batch_size_for(dataset):
    return 1 if dataset == "Xinjiang" else 16


def command_for(dataset, candidate, stage, args, summary_path, selection_only):
    config = {**candidate["config"], "batch_size": batch_size_for(dataset)}
    command = [
        sys.executable, "-u", str(PROJECT_ROOT / "main.py"),
        "--dataset", dataset,
        "--epochs", str(stage["epochs"]),
        "--num_runs", str(stage["runs"]),
        "--patience", str(args.early_stopping_patience),
        "--selection_metric", "mae",
        "--validation_missing_types", *args.validation_types,
        "--validation_missing_ratios", *map(str, args.validation_ratios),
        "--validation_worst_weight", str(args.worst_condition_weight),
        "--ablation_name", candidate["name"] + "_" + stage["name"],
        "--output_dir", str(summary_path.parent),
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
    if selection_only:
        command.append("--selection_only")
    return command


def score_summary(summary, stability_weight):
    score = summary["validation"]["selection_score"]
    return float(score["mean"]) + stability_weight * float(score["std"])


def run_candidate(dataset, candidate, stage, args, root, manifest, manifest_path, selection_only):
    name = candidate["name"]
    summary_path = root / "trials" / name / (stage["name"] + ".json")
    record = manifest["candidates"][name].setdefault("stages", {}).get(stage["name"], {})
    if summary_path.is_file() and not args.rerun_completed:
        summary = json.loads(summary_path.read_text())
        record.update({
            "status": "completed", "summary_path": str(summary_path),
            "score": score_summary(summary, args.stability_weight),
        })
        manifest["candidates"][name]["stages"][stage["name"]] = record
        write_json(manifest_path, manifest)
        return record
    if record.get("status") == "failed" and not args.retry_failed:
        return record

    command = command_for(dataset, candidate, stage, args, summary_path, selection_only)
    record = {"status": "running", "summary_path": str(summary_path), "command": command}
    manifest["candidates"][name].setdefault("stages", {})[stage["name"]] = record
    write_json(manifest_path, manifest)
    print("\n" + "#" * 88, flush=True)
    print(f"{dataset} | {stage['name']} | {name} | runs={stage['runs']} epochs={stage['epochs']}", flush=True)
    print("Executing:", " ".join(command), flush=True)
    print("#" * 88, flush=True)
    if args.dry_run:
        record["status"] = "dry_run"
        write_json(manifest_path, manifest)
        return record
    try:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        summary = json.loads(summary_path.read_text())
        record.update({
            "status": "completed",
            "score": score_summary(summary, args.stability_weight),
            "validation": summary["validation"],
        })
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as error:
        record.update({"status": "failed", "error": str(error)})
    write_json(manifest_path, manifest)
    return record


def ranked(candidates, stage_name, manifest):
    completed = [
        candidate for candidate in candidates
        if manifest["candidates"][candidate["name"]].get("stages", {}).get(stage_name, {}).get("status") == "completed"
    ]
    return sorted(completed, key=lambda candidate: manifest["candidates"][candidate["name"]]["stages"][stage_name]["score"])


def run_region(dataset, args):
    root = Path(args.results_root) / dataset
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not args.rerun_completed:
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "description": "Robust multi-condition validation hyperparameter search.",
            "dataset": dataset,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "arguments": vars(args),
            "validation_suite": {
                "types": args.validation_types,
                "ratios": args.validation_ratios,
                "worst_condition_weight": args.worst_condition_weight,
            },
            "candidates": {candidate["name"]: {"config": candidate["config"], "stages": {}} for candidate in CANDIDATES},
        }
        write_json(manifest_path, manifest)

    stages = (
        {"name": "screen", "epochs": args.screen_epochs, "runs": args.screen_runs},
        {"name": "rerank", "epochs": args.rerank_epochs, "runs": args.rerank_runs},
        {"name": "confirm", "epochs": args.confirm_epochs, "runs": args.confirm_runs},
    )
    if args.dry_run:
        for candidate in CANDIDATES:
            print(" ".join(command_for(
                dataset, candidate, stages[0], args,
                root / "trials" / candidate["name"] / "screen.json", True,
            )), flush=True)
        return {"status": "dry_run"}

    started = time.time()
    for candidate in CANDIDATES:
        run_candidate(dataset, candidate, stages[0], args, root, manifest, manifest_path, selection_only=True)
    rerank_candidates = ranked(CANDIDATES, "screen", manifest)[:args.rerank_top_k]
    if not rerank_candidates:
        raise RuntimeError(f"No screening candidate completed for {dataset}.")
    for candidate in rerank_candidates:
        run_candidate(dataset, candidate, stages[1], args, root, manifest, manifest_path, selection_only=True)
    confirm_candidates = ranked(rerank_candidates, "rerank", manifest)[:args.finalists]
    if not confirm_candidates:
        raise RuntimeError(f"No reranking candidate completed for {dataset}.")
    for candidate in confirm_candidates:
        run_candidate(dataset, candidate, stages[2], args, root, manifest, manifest_path, selection_only=True)
    finalists = ranked(confirm_candidates, "confirm", manifest)
    if not finalists:
        raise RuntimeError(f"No confirmation candidate completed for {dataset}.")

    winner = finalists[0]
    ranking = {
        stage["name"]: [
            {"candidate": candidate["name"], "score": manifest["candidates"][candidate["name"]]["stages"][stage["name"]]["score"]}
            for candidate in ranked(CANDIDATES, stage["name"], manifest)
        ]
        for stage in stages
    }
    write_json(root / "ranking.json", ranking)
    final_path = root / "final_evaluation.json"
    best = {
        "dataset": dataset,
        "selection_metric": "robust_validation_mae",
        "selection_score": manifest["candidates"][winner["name"]]["stages"]["confirm"]["score"],
        "winner": winner["name"],
        "config": {**winner["config"], "batch_size": batch_size_for(dataset)},
        "ranked_finalists": [
            {"candidate": candidate["name"], "score": manifest["candidates"][candidate["name"]]["stages"]["confirm"]["score"]}
            for candidate in finalists
        ],
        "final_evaluation_path": str(final_path),
    }
    write_json(root / "best_config.json", best)
    manifest["best_candidate"] = winner["name"]
    manifest["elapsed_seconds"] = time.time() - started
    write_json(manifest_path, manifest)

    if not final_path.is_file() or args.rerun_completed:
        command = command_for(dataset, winner, stages[2], args, final_path, selection_only=False)
        print("\nRunning the locked winner on the full test suite.", flush=True)
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return {"status": "completed", "winner": winner["name"], "best_config_path": str(root / "best_config.json")}


def main():
    parser = argparse.ArgumentParser(description="Run the robust narrow HPO study for all four regions.")
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--results-root", default="results/robust_hyperparameter_search")
    parser.add_argument("--screen-epochs", type=int, default=45)
    parser.add_argument("--screen-runs", type=int, default=1)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--rerank-epochs", type=int, default=100)
    parser.add_argument("--rerank-runs", type=int, default=3)
    parser.add_argument("--finalists", type=int, default=2)
    parser.add_argument("--confirm-epochs", type=int, default=120)
    parser.add_argument("--confirm-runs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--validation-types", nargs="+", choices=VALIDATION_TYPES, default=list(VALIDATION_TYPES))
    parser.add_argument("--validation-ratios", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    parser.add_argument("--worst-condition-weight", type=float, default=0.5)
    parser.add_argument("--stability-weight", type=float, default=0.1)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if min(args.screen_epochs, args.screen_runs, args.rerank_top_k, args.rerank_epochs, args.rerank_runs, args.finalists, args.confirm_epochs, args.confirm_runs) < 1:
        parser.error("All epoch counts and run counts must be positive.")
    if args.finalists > args.rerank_top_k or args.rerank_top_k > len(CANDIDATES):
        parser.error("Require --finalists <= --rerank-top-k <= number of candidates.")
    if not args.validation_ratios or any(not 0 < ratio < 1 for ratio in args.validation_ratios):
        parser.error("--validation-ratios must be in (0, 1).")
    if args.worst_condition_weight < 0 or args.stability_weight < 0:
        parser.error("Weights must be non-negative.")

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    outcomes = {}
    for dataset in datasets:
        print("\n" + "=" * 88, flush=True)
        print(f"Starting robust hyperparameter search for {dataset}", flush=True)
        print("=" * 88, flush=True)
        try:
            outcomes[dataset] = run_region(dataset, args)
        except (RuntimeError, subprocess.CalledProcessError) as error:
            outcomes[dataset] = {"status": "failed", "error": str(error)}
    write_json(Path(args.results_root) / "all_regions_summary.json", {"outcomes": outcomes})
    if any(outcome["status"] == "failed" for outcome in outcomes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
