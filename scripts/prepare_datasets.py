#!/usr/bin/env python3
"""Convert raw hourly ``obs.npy``/``era5.npy`` files into train/val/test arrays.

Raw arrays must have shape ``[stations, time, features]`` and be ordered as
TMP, DEW, U_W, V_W, PRE. The script never normalizes or imputes values; the
training loader fits scalers on the training split only.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def hour_index(origin: datetime, value: datetime) -> int:
    seconds = (value - origin).total_seconds()
    if seconds % 3600:
        raise ValueError(f"Timestamp is not aligned to one hour: {value}")
    return int(seconds // 3600)


def slice_inclusive(array, start: int, end: int):
    return array[:, start:end + 1, :]


def process_dataset(name: str, cfg: dict, raw_root: Path, output_root: Path):
    source = raw_root / name
    destination = output_root / name
    obs_path = source / "obs.npy"
    if not obs_path.exists():
        raise FileNotFoundError(f"Missing {obs_path}")

    obs = np.load(obs_path, mmap_mode="r")
    if obs.ndim != 3 or obs.shape[-1] != 5:
        raise ValueError(f"{obs_path} must have shape [stations,time,5], got {obs.shape}")
    origin = parse_time(cfg["raw_start"])
    overall_start = parse_time(cfg.get("slice_start", cfg["raw_start"]))
    overall_end = parse_time(cfg["slice_end"])
    begin = hour_index(origin, overall_start)
    finish = hour_index(origin, overall_end)
    sliced_obs = slice_inclusive(obs, begin, finish)

    era5_path = source / "era5.npy"
    sliced_era5 = None
    if era5_path.exists():
        era5 = np.load(era5_path, mmap_mode="r")
        if era5.shape != obs.shape:
            raise ValueError(f"ERA5 shape {era5.shape} does not match observations {obs.shape}")
        sliced_era5 = slice_inclusive(era5, begin, finish)

    splits = {
        "train": (cfg["train_start"], cfg["train_end"]),
        "val": (cfg["val_start"], cfg["val_end"]),
        "test": (cfg["test_start"], cfg["slice_end"]),
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "obs").mkdir(exist_ok=True)
    if sliced_era5 is not None:
        (destination / "era5").mkdir(exist_ok=True)

    metadata = {"dataset": name, "variables": ["TMP", "DEW", "U_W", "V_W", "PRE"],
                "raw_start": cfg["raw_start"], "has_era5": sliced_era5 is not None,
                "splits": {}}
    for split, (start_text, end_text) in splits.items():
        left = hour_index(overall_start, parse_time(start_text))
        right = hour_index(overall_start, parse_time(end_text))
        obs_part = np.asarray(slice_inclusive(sliced_obs, left, right), dtype=np.float32)
        np.save(destination / "obs" / f"{split}.npy", obs_part)
        if sliced_era5 is not None:
            era5_part = np.asarray(slice_inclusive(sliced_era5, left, right), dtype=np.float32)
            np.save(destination / "era5" / f"{split}.npy", era5_part)
        metadata["splits"][split] = {"start": start_text, "end": end_text,
                                      "shape": list(obs_part.shape)}
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{name}: wrote {destination} ({'with' if sliced_era5 is not None else 'without'} ERA5)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Directory containing one subdirectory per dataset.")
    parser.add_argument("--output-root", type=Path, default=Path("processed_data"))
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON file with dataset timestamps; see configs/datasets.example.json.")
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    names = args.datasets or list(config)
    for name in names:
        if name not in config:
            raise KeyError(f"No configuration for {name}")
        process_dataset(name, config[name], args.raw_root, args.output_root)


if __name__ == "__main__":
    main()
