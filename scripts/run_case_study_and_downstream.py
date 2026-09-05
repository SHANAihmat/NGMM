#!/usr/bin/env python3
"""Train five imputers once per region, cache qualitative cases, and test forecasting.

The script deliberately keeps qualitative-case selection independent of model
outputs. It selects the most dynamically active test station/window using only
ground-truth first differences, applies fixed PyGrinder masks, and saves the
raw-scale series used by the plotting script. The downstream task forecasts
one or more configurable horizons after imputation of a fixed context window.
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import SpatioTemporalDataset, StandardScaler, get_dataloaders
from model import PhysicsInformedReconstructor, physics_loss_function
from main import compute_innovation_scales
from baselines.fsdi_rerun_all import (
    build_model_class,
    fsdi_config,
    load_fsdi_base,
    prepare_batch,
    train_model as train_fsdi,
)
from pypots.imputation import HELIX, TEFN


DATASETS = ("France", "Australia", "Zhejiang", "Xinjiang")
METHODS = ("TEFN", "HELIX", "FSDI", "NGMM", "NGMM+ERA5")
CASE_TYPES = ("point_mcar", "seq_mcar", "block_2d")
VARIABLES = ("TMP", "DEW", "U_W", "V_W", "PRE")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def raw_splits(dataset):
    base = ROOT / "processed_data" / dataset / "obs"
    train = np.load(base / "train.npy").astype(np.float32)
    val = np.load(base / "val.npy").astype(np.float32)
    test = np.load(base / "test.npy").astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(train)
    return tuple(scaler.transform(x).astype(np.float32) for x in (train, val, test)), scaler


def era5_splits(dataset):
    base = ROOT / "processed_data" / dataset / "era5"
    train = np.load(base / "train.npy").astype(np.float32)
    val = np.load(base / "val.npy").astype(np.float32)
    test = np.load(base / "test.npy").astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(train)
    return tuple(scaler.transform(x).astype(np.float32) for x in (train, val, test)), scaler


def to_pypots(loader):
    xs, originals = [], []
    for masked, mask, _, gt in loader:
        x = np.where(mask.numpy() == 1, masked.numpy(), np.nan)
        xs.append(np.transpose(x, (0, 2, 1, 3)))
        originals.append(np.transpose(gt.numpy(), (0, 2, 1, 3)))
    x = np.concatenate(xs, axis=0)
    original = np.concatenate(originals, axis=0)
    b, length, nodes, features = x.shape
    return {
        "X": x.reshape(b, length, nodes * features),
        "X_ori": original.reshape(b, length, nodes * features),
        "nodes": nodes,
        "features": features,
    }


def pypots_predict(model, masked, mask):
    b, nodes, length, features = masked.shape
    x = np.where(mask.cpu().numpy() == 1, masked.cpu().numpy(), np.nan)
    x = np.transpose(x, (0, 2, 1, 3)).reshape(b, length, nodes * features)
    output = np.asarray(model.impute({"X": x}))
    return torch.as_tensor(output.reshape(b, length, nodes, features).transpose(0, 2, 1, 3))


def fsdi_predict(model, masked, mask, gt, n_samples):
    raw = (masked, mask, torch.empty(0), gt)
    batch = prepare_batch(raw, evaluation=True)
    with torch.no_grad():
        samples, _, _, _, _ = model.evaluate(batch, n_samples)
    # Native FSDI samples: [B, samples, target_dim, time].
    flat = samples.median(dim=1).values.permute(0, 2, 1)
    b, nodes, length, features = gt.shape
    return flat.reshape(b, length, nodes, features).permute(0, 2, 1, 3).cpu()


def ngmm_predict(model, masked, mask, era5, device):
    with torch.no_grad():
        return model(
            masked.to(device), mask.to(device), era5.to(device) if era5.numel() else era5
        ).cpu()


def complete(prediction, gt, mask):
    """Preserve observed inputs so every downstream forecaster sees a completed record."""
    return mask.cpu() * gt.cpu() + (1.0 - mask.cpu()) * prediction.cpu()


def select_case(test, context):
    """Return a reproducible (window, station) determined only from ground truth."""
    starts = np.arange(0, test.shape[1] - context + 1, context)
    energies = []
    for start in starts:
        delta = np.diff(test[:, start : start + context], axis=1)
        energies.append(np.abs(delta).sum(axis=(1, 2)))
    energies = np.stack(energies)
    window, station = np.unravel_index(np.argmax(energies), energies.shape)
    return int(window), int(station), int(starts[window])


def make_case_sample(test_norm, era_norm, missing_type, ratio, context, window_index):
    ds = SpatioTemporalDataset(
        test_norm, era_norm, context, stride=context, mode="test",
        missing_ratio=ratio, missing_type=missing_type,
    )
    return tuple(value.unsqueeze(0) for value in ds[window_index])


def ngmm_config(dataset, use_era5):
    config_path = ROOT / "results" / "robust_hyperparameter_search" / dataset / "best_config.json"
    config = json.loads(config_path.read_text())["config"]
    # ERA5 runs use the stable direct-training settings documented in the project.
    if use_era5:
        config.update({"hidden_dim": 256, "num_layers": 3, "num_heads": 4,
                       "kernel_size": 3, "lr": 5e-4, "batch_size": 16,
                       "temporal_weight": 0.1, "wind_vector_weight": 0.025})
    config["use_era5"] = int(use_era5)
    return config


def train_ngmm(dataset, train_loader, val_loader, device, epochs, use_era5):
    cfg = ngmm_config(dataset, use_era5)
    model = PhysicsInformedReconstructor(
        num_features=5, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], kernel_size=cfg["kernel_size"],
        use_era5=use_era5, feature_weights=cfg["feature_weights"],
        use_temporal_smooth=True, use_time_attn=True, use_var_attn=True,
        temporal_fusion_mode="gated_cascade",
    ).to(device)
    (_, scaler) = raw_splits(dataset)
    delta_scales, wind_scale = compute_innovation_scales(
        str(ROOT / "processed_data" / dataset), scaler, cfg.get("temporal_scale_floor", 0.05)
    )
    loss_kwargs = dict(
        beta=cfg["beta"], missing_penalty=cfg["missing_penalty"],
        temporal_weight=cfg["temporal_weight"], wind_vector_weight=cfg["wind_vector_weight"],
        delta_scales=torch.tensor(delta_scales, device=device),
        wind_delta_scale=torch.tensor(wind_scale, device=device), pointwise_loss=cfg["pointwise_loss"],
        temporal_feature_weights=cfg["temporal_feature_weights"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, stale, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        for masked, mask, era, gt in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(masked.to(device), mask.to(device), era.to(device))
            loss = physics_loss_function(pred, gt.to(device), mask.to(device), model.feature_weights, **loss_kwargs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip_norm", 1.0))
            optimizer.step()
        model.eval()
        error, count = 0.0, 0.0
        with torch.no_grad():
            for masked, mask, era, gt in val_loader:
                pred = model(masked.to(device), mask.to(device), era.to(device))
                missing = 1.0 - mask.to(device)
                error += (torch.abs(pred - gt.to(device)) * missing).sum().item()
                count += missing.sum().item()
        score = error / max(count, 1.0)
        if score < best:
            best, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 20:
                break
    model.load_state_dict(best_state)
    return model, {"validation_mae_normalized": best, "config": cfg, "epochs_run": epoch + 1}


def fsdi_args(epochs, device):
    return SimpleNamespace(
        epochs=epochs, lr=1e-3, validation_every=10, max_train_batches=None,
        max_val_batches=None, layers=4, channels=64, heads=8,
        diffusion_embedding_dim=128, diffusion_steps=50, disable_freq_noise=False,
        device=device,
    )


@dataclass
class TrainedMethod:
    name: str
    model: object
    metadata: dict


def train_methods(dataset, args, device):
    data_dir = ROOT / "processed_data" / dataset
    train_loader, val_loader, _, _, _ = get_dataloaders(
        dataset_dir=str(data_dir), seq_len=args.context, train_stride=args.context // 2,
        test_stride=args.context, batch_size=args.batch_size, train_missing_ratio=0.2,
        test_missing_ratio=0.2, missing_type="point_mcar",
    )
    train_pypots, val_pypots = to_pypots(train_loader), to_pypots(val_loader)
    train_set = {"X": train_pypots["X"]}
    val_set = {"X": val_pypots["X"], "X_ori": val_pypots["X_ori"]}
    feature_dim = train_pypots["X"].shape[-1]
    methods = {}
    for name, cls in (("TEFN", TEFN), ("HELIX", HELIX)):
        if name not in args.methods:
            continue
        seed_everything(args.seed)
        model = cls(n_steps=args.context, n_features=feature_dim, epochs=args.epochs,
                    batch_size=args.pypots_batch_size, patience=args.patience,
                    device=device, saving_path=None, verbose=True)
        model.fit(train_set, val_set)
        methods[name] = TrainedMethod(name, model, {"epochs": args.epochs})
    if "FSDI" in args.methods:
        seed_everything(args.seed)
        fsdi_class = build_model_class(load_fsdi_base())
        fsdi = fsdi_class(target_dim=feature_dim, config=fsdi_config(fsdi_args(args.epochs, device), args.context), device=device).to(device)
        best_loss = train_fsdi(fsdi, train_loader, val_loader, fsdi_args(args.epochs, device))
        methods["FSDI"] = TrainedMethod("FSDI", fsdi, {"validation_loss": best_loss})
    for name, use_era5 in (("NGMM", False), ("NGMM+ERA5", True)):
        if name not in args.methods:
            continue
        seed_everything(args.seed)
        model, meta = train_ngmm(dataset, train_loader, val_loader, device, args.epochs, use_era5)
        methods[name] = TrainedMethod(name, model, meta)
    return methods


def predict_method(trained, masked, mask, era, gt, device, n_samples):
    if trained.name in ("TEFN", "HELIX"):
        return pypots_predict(trained.model, masked, mask)
    if trained.name == "FSDI":
        return fsdi_predict(trained.model, masked, mask, gt, n_samples)
    era = era if trained.name == "NGMM+ERA5" else torch.empty(0)
    return ngmm_predict(trained.model, masked, mask, era, device)


def inverse(scaler, x):
    shape = x.shape
    return scaler.inverse_transform(x.reshape(-1, shape[-1])).reshape(shape)


def cache_cases(dataset, methods, test_norm, era_norm, scaler, args, out_dir, case):
    case_dir = out_dir / "predictions" / dataset
    case_dir.mkdir(parents=True, exist_ok=True)
    window, station, start = case
    for missing_type in CASE_TYPES:
        masked, mask, era, gt = make_case_sample(
            test_norm, era_norm, missing_type, args.case_ratio, args.context, window
        )
        record = {
            "ground_truth": inverse(scaler, gt.numpy()[0, station]),
            "observed": inverse(scaler, masked.numpy()[0, station]),
            "mask": mask.numpy()[0, station],
            "station_index": np.array(station), "window_index": np.array(window),
            "start_index": np.array(start),
        }
        for name, trained in methods.items():
            pred = predict_method(trained, masked, mask, era, gt, args.device, args.fsdi_samples)
            record[name] = inverse(scaler, complete(pred, gt, mask).numpy()[0, station])
        np.savez_compressed(case_dir / f"{missing_type}_ratio{args.case_ratio:g}.npz", **record)


def forecasting_windows(data, era, context, horizon, stride, ratio, missing_type):
    """Create leakage-free contexts and targets for a fixed forecast horizon."""
    if horizon < 1 or horizon >= data.shape[1] - context:
        raise ValueError("forecast horizon must be positive and shorter than the available sequence.")
    # Exclude the horizon and all future values from the imputer input. For a
    # window beginning at t, the target is at t + context + horizon - 1.
    ds = SpatioTemporalDataset(data[:, :-horizon], era[:, :-horizon] if era is not None else None,
                               context, stride=stride, mode="test", missing_ratio=ratio,
                               missing_type=missing_type)
    batches, targets = [], []
    for index in range(len(ds)):
        masked, mask, aux, gt = ds[index]
        batches.append((masked, mask, aux, gt))
        targets.append(data[:, index * stride + context + horizon - 1])
    return batches, np.asarray(targets, dtype=np.float32)


def impute_windows(trained, batches, device, n_samples, batch_size):
    completed = []
    for first in range(0, len(batches), batch_size):
        group = batches[first : first + batch_size]
        masked = torch.stack([x[0] for x in group])
        mask = torch.stack([x[1] for x in group])
        era = torch.stack([x[2] for x in group]) if group[0][2].numel() else torch.empty(0)
        gt = torch.stack([x[3] for x in group])
        pred = predict_method(trained, masked, mask, era, gt, device, n_samples)
        completed.append(complete(pred, gt, mask).numpy())
    return np.concatenate(completed, axis=0)


def downstream_forecasting(dataset, methods, train, test, train_era, test_era, scaler, args, out_dir):
    rows = []
    for horizon in args.forecast_horizons:
        train_batches, train_y = forecasting_windows(train, train_era, args.context, horizon, args.forecast_stride, args.downstream_ratio, args.downstream_missing_type)
        test_batches, test_y = forecasting_windows(test, test_era, args.context, horizon, args.forecast_stride, args.downstream_ratio, args.downstream_missing_type)
        for name, trained in methods.items():
            train_x = impute_windows(trained, train_batches, args.device, args.fsdi_samples, args.inference_batch_size)
            test_x = impute_windows(trained, test_batches, args.device, args.fsdi_samples, args.inference_batch_size)
            x_train = train_x.reshape(len(train_x), -1)
            x_test = test_x.reshape(len(test_x), -1)
            y_train = train_y.reshape(len(train_y), -1)
            y_test = test_y.reshape(len(test_y), -1)
            forecaster = Ridge(alpha=args.ridge_alpha, fit_intercept=True)
            forecaster.fit(x_train, y_train)
            pred = forecaster.predict(x_test).reshape(test_y.shape)
            pred_raw, y_raw = inverse(scaler, pred), inverse(scaler, test_y)
            error = pred_raw - y_raw
            forecast_dir = out_dir / "downstream_forecasts" / dataset
            forecast_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                forecast_dir / f"{name.replace('+', '_plus_')}_h{horizon}.npz",
                target=y_raw, prediction=pred_raw, context_imputed=inverse(scaler, test_x),
            )
            rows.append({"dataset": dataset, "method": name, "missing_type": args.downstream_missing_type,
                         "missing_ratio": args.downstream_ratio, "horizon_hours": horizon,
                         "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
                         "n_train": int(len(train_x)), "n_test": int(len(test_x))})
    return rows


def write_csv(path, rows):
    columns = list(rows[0])
    path.write_text(",".join(columns) + "\n" + "\n".join(
        ",".join(str(row[key]) for key in columns) for row in rows
    ) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--context", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16, help="NGMM/data-loader batch size")
    parser.add_argument("--pypots-batch-size", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--fsdi-samples", type=int, default=10)
    parser.add_argument("--case-ratio", type=float, default=0.3)
    parser.add_argument("--downstream-ratio", type=float, default=0.3)
    parser.add_argument("--downstream-missing-type", default="block_2d",
                        choices=("point_mcar", "point_mar", "point_mnar_x", "seq_mcar", "block_2d"))
    parser.add_argument("--forecast-stride", type=int, default=6)
    parser.add_argument("--forecast-horizons", nargs="+", type=int, default=[6, 12],
                        help="Positive leakage-free forecast horizons in hours, e.g. 1 3 6 12 24.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS,
                        help="Imputers included in both the cached case study and downstream task.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true", help="Validate data, ERA5 alignment, and deterministic case selection without training.")
    return parser.parse_args()


def main():
    args = parse_args()
    if any(horizon < 1 for horizon in args.forecast_horizons):
        raise ValueError("--forecast-horizons must contain positive integers.")
    args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Run this pipeline on the GPU server; do not silently fall back to CPU.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": {"case_ratio": args.case_ratio, "case_types": CASE_TYPES,
        "case_selection": "maximum ground-truth standardized first-difference energy",
        "downstream": {"missing_type": args.downstream_missing_type, "missing_ratio": args.downstream_ratio,
                       "context_hours": args.context, "forecast_horizons_hours": args.forecast_horizons,
                       "forecaster": "ridge", "methods": args.methods}},
        "variables": VARIABLES, "device": str(args.device), "regions": {}}
    all_rows = []
    for dataset in args.datasets:
        print(f"\n===== {dataset} =====", flush=True)
        seed_everything(args.seed)
        (train, _val, test), scaler = raw_splits(dataset)
        (era_train, _era_val, era_test), _era_scaler = era5_splits(dataset)
        case = select_case(test, args.context)
        region_manifest = {"case": {"window_index": case[0], "station_index": case[1], "start_index": case[2]},
                           "aws_shapes": {"train": list(train.shape), "test": list(test.shape)},
                           "era5_shapes": {"train": list(era_train.shape), "test": list(era_test.shape)}}
        if not args.dry_run:
            methods = train_methods(dataset, args, args.device)
            cache_cases(dataset, methods, test, era_test, scaler, args, args.output_dir, case)
            rows = downstream_forecasting(dataset, methods, train, test, era_train, era_test, scaler, args, args.output_dir)
            all_rows.extend(rows)
            region_manifest["methods"] = {name: method.metadata for name, method in methods.items()}
        manifest["regions"][dataset] = region_manifest
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if all_rows:
            write_csv(args.output_dir / "downstream_metrics.csv", all_rows)
    print(f"Completed. Cached predictions and metrics: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
