#!/usr/bin/env python3
"""Dependency and tensor-shape smoke test; does not require a dataset or GPU."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import PhysicsInformedReconstructor, physics_loss_function


def check(use_era5: bool):
    batch, stations, length, features, hidden = 2, 3, 24, 5, 96
    model = PhysicsInformedReconstructor(
        num_features=features, hidden_dim=hidden, num_layers=2,
        num_heads=4, kernel_size=3, use_era5=use_era5,
    )
    x = torch.randn(batch, stations, length, features)
    mask = (torch.rand_like(x) > 0.2).float()
    era5 = torch.randn_like(x) if use_era5 else torch.empty(0)
    prediction = model(x, mask, era5)
    assert prediction.shape == x.shape
    loss = physics_loss_function(
        prediction, x, mask, model.feature_weights,
        beta=0.1, missing_penalty=2.0, temporal_weight=0.1,
        wind_vector_weight=0.025,
    )
    assert torch.isfinite(loss), loss
    print(f"use_era5={use_era5}: output={tuple(prediction.shape)}, loss={loss.item():.4f}")


if __name__ == "__main__":
    check(False)
    check(True)
    print("NGMM smoke test passed.")
