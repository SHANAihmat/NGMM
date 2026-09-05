import argparse
import json
import os
import torch
import torch.optim as optim
import numpy as np
import random
import time   # 🌟 引入 time 模块用于计时
from collections import defaultdict
import torch.cuda.amp as amp 

from dataset import get_dataloaders
from model import PhysicsInformedReconstructor, physics_loss_function, vanilla_loss_function

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

TEST_MISSING_TYPES = ["point_mcar", "point_mar", "point_mnar_x", "seq_mcar", "block_2d"]
TEST_MISSING_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def compute_innovation_scales(dataset_dir, obs_scaler, floor=0.05):
    """Estimate region-specific temporal scales from the training split only."""
    train_obs = np.load(os.path.join(dataset_dir, 'obs', 'train.npy'))
    normalized = obs_scaler.transform(train_obs).astype(np.float32, copy=False)
    deltas = np.diff(normalized, axis=1)

    median = np.nanmedian(deltas, axis=(0, 1), keepdims=True)
    mad = np.nanmedian(np.abs(deltas - median), axis=(0, 1))
    delta_scales = np.clip(1.4826 * mad, floor, None).astype(np.float32)

    wind_norm = np.linalg.vector_norm(deltas[..., 2:4], axis=-1)
    wind_median = np.nanmedian(wind_norm)
    wind_mad = np.nanmedian(np.abs(wind_norm - wind_median))
    wind_scale = float(max(1.4826 * wind_mad, floor))
    return delta_scales, wind_scale


def loss_kwargs_from_args(args, delta_scales, wind_delta_scale):
    kwargs = {
        'beta': args.beta,
        'missing_penalty': args.missing_penalty,
        'temporal_weight': args.temporal_weight,
        'wind_vector_weight': args.wind_vector_weight,
        'pointwise_loss': args.pointwise_loss,
        'temporal_feature_weights': args.temporal_feature_weights,
    }
    if args.adaptive_temporal_scale == 1:
        kwargs['delta_scales'] = delta_scales
        kwargs['wind_delta_scale'] = wind_delta_scale
    return kwargs


def evaluate_validation(model, val_loader, obs_scaler, device, loss_kwargs):
    """Evaluate the training loss and raw-scale masked reconstruction metrics."""
    raw_scale = torch.as_tensor(
        obs_scaler.std.reshape(-1), device=device, dtype=torch.float32
    ).view(1, 1, 1, -1)
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_mse = torch.zeros((), device=device, dtype=torch.float32)
    total_mae = torch.zeros((), device=device, dtype=torch.float32)
    count = torch.zeros((), device=device, dtype=torch.float32)
    with torch.no_grad():
        for masked_obs, mask, era5, gt_obs in val_loader:
            masked_obs, mask = masked_obs.to(device), mask.to(device)
            era5, gt_obs = era5.to(device), gt_obs.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                pred_obs = model(masked_obs, mask, era5)
                loss = physics_loss_function(
                    pred_obs, gt_obs, mask, model.feature_weights, **loss_kwargs
                )

            missing = 1.0 - mask.float()
            raw_diff = (pred_obs.float() - gt_obs.float()) * missing * raw_scale
            total_loss += loss.detach()
            total_mse += raw_diff.square().sum()
            total_mae += raw_diff.abs().sum()
            count += missing.sum()

    return {
        'loss': (total_loss / max(len(val_loader), 1)).item(),
        'mae': (total_mae / count.clamp_min(1.0)).item(),
        'mse': (total_mse / count.clamp_min(1.0)).item(),
    }


def evaluate_validation_suite(model, validation_loaders, obs_scaler, device, loss_kwargs, metric, worst_weight):
    """Score one checkpoint over several deterministic validation masks.

    The suite is used only for model selection. It prevents a candidate from
    winning solely because it is specialized for one point-MCAR condition.
    """
    condition_metrics = {
        name: evaluate_validation(model, loader, obs_scaler, device, loss_kwargs)
        for name, loader in validation_loaders
    }
    values = [metrics[metric] for metrics in condition_metrics.values()]
    aggregate = {
        name: float(np.mean([metrics[name] for metrics in condition_metrics.values()]))
        for name in ('loss', 'mae', 'mse')
    }
    aggregate['selection_score'] = float(np.mean(values) + worst_weight * max(values))
    return aggregate, condition_metrics


def train_and_evaluate_single_run(args, device, run_idx):
    print(f"\n{'-'*20} Starting Run {run_idx + 1}/{args.num_runs} {'-'*20}")
    DATASET_DIR = os.path.join(os.path.abspath(args.data_root), args.dataset)

    train_loader, val_loader, _, obs_scaler, era5_scaler = get_dataloaders(
        dataset_dir=DATASET_DIR, seq_len=args.seq_len,
        train_stride=args.train_stride or args.seq_len // 2, test_stride=args.seq_len,
        batch_size=args.batch_size,
        train_missing_ratio=args.train_missing_ratio, test_missing_ratio=args.val_missing_ratio,
        missing_type=args.train_missing_type,
        use_augmentation=(args.use_augmentation == 1),
        aug_noise_std=args.aug_noise_std,
        aug_scale_range=(1.0 - args.aug_scale_jitter, 1.0 + args.aug_scale_jitter),
        aug_shift_range=(-args.aug_max_shift, args.aug_max_shift + 1),
        aug_dropout_rate=args.aug_dropout_rate,
        aug_prob=args.aug_prob,
    )
    delta_scales, wind_delta_scale = compute_innovation_scales(
        DATASET_DIR, obs_scaler, floor=args.temporal_scale_floor
    )

    validation_loaders = [(f'{args.train_missing_type}_ratio{args.val_missing_ratio:g}', val_loader)]
    if args.validation_missing_types is not None:
        loader_cache = {(args.train_missing_type, args.val_missing_ratio): val_loader}
        validation_loaders = []
        for missing_type in args.validation_missing_types:
            for missing_ratio in args.validation_missing_ratios:
                key = (missing_type, missing_ratio)
                if key not in loader_cache:
                    _, suite_loader, _, _, _ = get_dataloaders(
                        dataset_dir=DATASET_DIR,
                        seq_len=args.seq_len,
                        train_stride=args.train_stride or args.seq_len // 2,
                        test_stride=args.seq_len,
                        batch_size=args.batch_size,
                        train_missing_ratio=args.train_missing_ratio,
                        test_missing_ratio=missing_ratio,
                        missing_type=missing_type,
                    )
                    loader_cache[key] = suite_loader
                validation_loaders.append((f'{missing_type}_ratio{missing_ratio:g}', loader_cache[key]))

    actual_use_era5 = (args.use_era5 == 1) and (era5_scaler is not None)
    if (args.use_era5 == 1) and not actual_use_era5:
        print("⚠️ Warning: `--use_era5 1` was requested, but no ERA5 data was found. Falling back to without_ERA5 mode.")

    # 🌟 初始化模型
    model = PhysicsInformedReconstructor(
        num_features=5, 
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers,
        use_era5=actual_use_era5,
        num_heads=args.num_heads,
        kernel_size=args.kernel_size,
        feature_weights=args.feature_weights,
        use_temporal_smooth=(args.use_temporal_smooth == 1),
        use_time_attn=(args.use_time_attn == 1),
        use_var_attn=(args.use_var_attn == 1),
        temporal_fusion_mode=args.temporal_fusion_mode,
    ).to(device)
    loss_kwargs = loss_kwargs_from_args(args, delta_scales, wind_delta_scale)
    if args.adaptive_temporal_scale == 1:
        loss_kwargs['delta_scales'] = torch.as_tensor(
            delta_scales, device=device, dtype=torch.float32
        )
        loss_kwargs['wind_delta_scale'] = torch.tensor(
            wind_delta_scale, device=device, dtype=torch.float32
        )

    # 🌟 统计模型参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Initialized. Trainable Parameters: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.scheduler_factor,
        patience=args.scheduler_patience, min_lr=args.min_lr,
    )
    amp_enabled = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    best_val_score = float('inf')
    best_val_metrics = None
    best_val_conditions = None
    patience_counter = 0
    checkpoint_path = f"temp_best_model_{args.dataset}_{os.getpid()}_run{run_idx}.pt"

    # 🌟 用于累计所有 epoch 的纯训练时间
    total_train_time = 0.0
    actual_epochs_run = 0

    for epoch in range(1, args.epochs + 1):
        actual_epochs_run += 1
        
        # 🌟 记录当前 epoch 的开始时间
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        epoch_start_time = time.time()
        
        model.train()
        total_train_loss = torch.zeros((), device=device, dtype=torch.float32)
        
        for masked_obs, mask, era5, gt_obs in train_loader:
            masked_obs, mask = masked_obs.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            era5, gt_obs = era5.to(device, non_blocking=True), gt_obs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred_obs = model(masked_obs, mask, era5)
                loss = physics_loss_function(
                    pred_obs, gt_obs, mask, model.feature_weights, **loss_kwargs
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.detach()
            
        # 🌟 记录当前 epoch 的结束时间，并累加到总时间中 (仅统计纯训练时间，不含验证时间)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        epoch_end_time = time.time()
        total_train_time += (epoch_end_time - epoch_start_time)
            
        avg_train_loss = (total_train_loss / len(train_loader)).item()

        model.eval()
        if args.validation_missing_types is None:
            val_metrics = evaluate_validation(model, val_loader, obs_scaler, device, loss_kwargs)
            val_score = val_metrics[args.selection_metric]
            val_conditions = None
        else:
            val_metrics, val_conditions = evaluate_validation_suite(
                model, validation_loaders, obs_scaler, device, loss_kwargs,
                args.selection_metric, args.validation_worst_weight,
            )
            val_score = val_metrics['selection_score']
        scheduler.step(val_score)

        if val_score < best_val_score:
            best_val_score = val_score
            best_val_metrics = val_metrics
            best_val_conditions = val_conditions
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
            print(
                f"  [Epoch {epoch:03d}/{args.epochs}] Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val MAE: {val_metrics['mae']:.4f} | "
                f"Val MSE: {val_metrics['mse']:.4f} | Selection: {val_score:.4f}  [Best Model Saved]"
            )
        else:
            patience_counter += 1
            print(
                f"  [Epoch {epoch:03d}/{args.epochs}] Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val MAE: {val_metrics['mae']:.4f} | "
                f"Val MSE: {val_metrics['mse']:.4f} | Selection: {val_score:.4f}  (Patience: {patience_counter}/{args.patience})"
            )
            if patience_counter >= args.patience:
                print(f"  ->  Early stopping at epoch {epoch}")
                break

    # 🌟 计算平均每个 epoch 的训练时长
    avg_time_per_epoch = total_train_time / actual_epochs_run if actual_epochs_run > 0 else 0

    print(f"  -> Training finished. Starting comprehensive inference...")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    run_results = {}
    
    # 🌟 记录总推理时长
    total_inference_time = 0.0
    num_conditions_tested = 0
    
    if not args.selection_only:
        for m_type in TEST_MISSING_TYPES:
            for m_ratio in TEST_MISSING_RATIOS:
                cond_name = f"{m_type}_ratio{m_ratio}"
                _, _, test_loader, _, _ = get_dataloaders(
                    dataset_dir=DATASET_DIR, seq_len=args.seq_len,
                    train_stride=args.seq_len, test_stride=args.seq_len,
                    batch_size=args.batch_size,
                    train_missing_ratio=args.train_missing_ratio, test_missing_ratio=m_ratio, missing_type=m_type
                )
            
                t_mse, t_mae, count = 0.0, 0.0, 0
            
                # 🌟 单个测试集的推理计时
                start_inf_time = time.time()
            
                with torch.no_grad():
                    for masked_obs, mask, era5, gt_obs in test_loader:
                        masked_obs, mask = masked_obs.to(device), mask.to(device)
                        era5, gt_obs = era5.to(device), gt_obs.to(device)
                        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                            pred_obs = model(masked_obs, mask, era5)
                        
                        final_imputed = pred_obs * (1.0 - mask) + masked_obs * mask
                    
                        final_imputed = final_imputed.float()
                        gt_obs = gt_obs.float()
                    
                        final_np = obs_scaler.inverse_transform(final_imputed.cpu().numpy())
                        gt_np    = obs_scaler.inverse_transform(gt_obs.cpu().numpy())
                        missing_mask = 1.0 - mask.cpu().numpy()
                    
                        diff = (final_np - gt_np) * missing_mask
                        t_mse += (diff ** 2).sum()
                        t_mae += np.abs(diff).sum()
                        count += missing_mask.sum()
                    
                # 🌟 记录推理耗时
                end_inf_time = time.time()
                total_inference_time += (end_inf_time - start_inf_time)
                num_conditions_tested += 1

                avg_mse = t_mse / count if count > 0 else 0
                avg_mae = t_mae / count if count > 0 else 0
                avg_rmse = np.sqrt(avg_mse)
            
                run_results[cond_name] = {"mse": avg_mse, "mae": avg_mae, "rmse": avg_rmse}
            
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # 🌟 计算每次条件测试的平均推理耗时
    avg_inference_time = total_inference_time / num_conditions_tested if num_conditions_tested > 0 else 0
    
    # 🌟 汇总运行的参数和效率指标
    meta_info = {
        "num_params": num_params,
        "avg_time_per_epoch": avg_time_per_epoch,
        "avg_inference_time": avg_inference_time,
        "validation": best_val_metrics,
        "validation_conditions": best_val_conditions,
        "delta_scales": delta_scales.tolist(),
        "wind_delta_scale": wind_delta_scale,
    }

    return run_results, meta_info  # 🌟 返回性能结果和效率指标

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f" CURRENT TRAINING DEVICE: {device.type.upper()} ")
    print(f" ERA5 MODE: {'ENABLED' if args.use_era5 == 1 else 'DISABLED'}")
    print(f"{'='*50}\n")
    
    all_metrics = defaultdict(lambda: defaultdict(list))
    all_meta = []  # 🌟 收集每次 run 的资源消耗情况
    
    for i in range(args.num_runs):
        set_seed(42 + i * 100)
        
        # 🌟 接收字典结果和元数据信息
        run_res, meta_info = train_and_evaluate_single_run(args, device, i)
        all_meta.append(meta_info)
        
        for cond_name, metrics in run_res.items():
            all_metrics[cond_name]["mse"].append(metrics["mse"])
            all_metrics[cond_name]["mae"].append(metrics["mae"])
            all_metrics[cond_name]["rmse"].append(metrics["rmse"])

    # 🌟 汇总运行时间和参数
    avg_params = all_meta[0]["num_params"]
    avg_train_time = np.mean([m["avg_time_per_epoch"] for m in all_meta])
    avg_inf_time = np.mean([m["avg_inference_time"] for m in all_meta])
    validation_metric_names = ['loss', 'mae', 'mse']
    if 'selection_score' in all_meta[0]['validation']:
        validation_metric_names.append('selection_score')
    validation_summary = {
        metric: {
            "mean": float(np.mean([m["validation"][metric] for m in all_meta])),
            "std": float(np.std([m["validation"][metric] for m in all_meta])),
        }
        for metric in validation_metric_names
    }
    validation_condition_summary = {}
    if all_meta[0]['validation_conditions'] is not None:
        for condition in all_meta[0]['validation_conditions']:
            validation_condition_summary[condition] = {
                metric: {
                    'mean': float(np.mean([m['validation_conditions'][condition][metric] for m in all_meta])),
                    'std': float(np.std([m['validation_conditions'][condition][metric] for m in all_meta])),
                }
                for metric in ('loss', 'mae', 'mse')
            }

    output_lines = [
        "======================================",
        f"Dataset: {args.dataset}",
        f"Use ERA5: {'True' if args.use_era5 == 1 else 'False'}",
        f"Use Temporal Smooth: {'True' if args.use_temporal_smooth == 1 else 'False'}", 
        f"Number of runs: {args.num_runs}",
        f"Hidden Dim: {args.hidden_dim} | Layers: {args.num_layers} | Heads: {args.num_heads}",
        f"Pointwise loss: {args.pointwise_loss} | Temporal innovation + u/v vector evolution",
        "Feature order: temperature, virtual temperature, u-wind, v-wind, pressure",
        f"Smooth L1 beta: {args.beta} | Missing Penalty: {args.missing_penalty} | Feature weights: {args.feature_weights}",
        f"Temporal Weight: {args.temporal_weight} | Wind Vector Weight: {args.wind_vector_weight}",
        f"Temporal feature weights: {args.temporal_feature_weights}",
        f"Adaptive Temporal Scale: {'True' if args.adaptive_temporal_scale == 1 else 'False'}",
        f"Validation Mask: {args.train_missing_type} ratio {args.val_missing_ratio}",
        (
            f"Validation (raw masked): MAE {validation_summary['mae']['mean']:.4f}"
            f"±{validation_summary['mae']['std']:.4f} | MSE {validation_summary['mse']['mean']:.4f}"
            f"±{validation_summary['mse']['std']:.4f}"
        ),
        "--------------------------------------",
        f"🌟 Performance & Efficiency Statistics:",
        f"   - Model Parameters       : {avg_params:,}",
        f"   - Avg Train Time / Epoch : {avg_train_time:.4f} seconds",
        f"   - Avg Inference Time     : {avg_inf_time:.4f} seconds (per test condition)",
        "======================================\n"
    ]
    
    print("\n" + "\n".join(output_lines))
    
    test_summary = {}
    if not args.selection_only:
        cond_keys = []
        for m_type in TEST_MISSING_TYPES:
            for m_ratio in TEST_MISSING_RATIOS:
                cond_keys.append(f"{m_type}_ratio{m_ratio}")

        for cond in cond_keys:
            mses = all_metrics[cond]["mse"]
            maes = all_metrics[cond]["mae"]
            rmses = all_metrics[cond]["rmse"]
            
            mse_m, mse_s = np.mean(mses), np.std(mses)
            mae_m, mae_s = np.mean(maes), np.std(maes)
            rmse_m, rmse_s = np.mean(rmses), np.std(rmses)
            test_summary[cond] = {
                "mse": {"mean": float(mse_m), "std": float(mse_s)},
                "mae": {"mean": float(mae_m), "std": float(mae_s)},
                "rmse": {"mean": float(rmse_m), "std": float(rmse_s)},
            }
            
            line = f"Condition [{cond:<25}] => MSE: {mse_m:.4f}±{mse_s:.4f} | MAE: {mae_m:.4f}±{mae_s:.4f} | RMSE: {rmse_m:.4f}±{rmse_s:.4f}"
            print(line)
            output_lines.append(line)

    era5_tag = "with_ERA5" if args.use_era5 == 1 else "without_ERA5"
    temporal_tag = "with_Temporal" if args.use_temporal_smooth == 1 else "no_Temporal"
    loss_tag = f"tw_{args.temporal_weight:g}_ww_{args.wind_vector_weight:g}"
    os.makedirs(args.output_dir, exist_ok=True)
    log_filename = os.path.join(
        args.output_dir,
        f"{args.dataset}_{era5_tag}_{args.ablation_name}_{temporal_tag}_{loss_tag}.txt",
    )
    
    with open(log_filename, "w") as f:
        f.write("\n".join(output_lines))

    summary = {
        "dataset": args.dataset,
        "selection_only": args.selection_only,
        "num_runs": args.num_runs,
        "model_config": {
            "num_params": int(avg_params),
            "use_era5": bool(args.use_era5),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "kernel_size": args.kernel_size,
            "use_temporal_smooth": bool(args.use_temporal_smooth),
            "use_time_attn": bool(args.use_time_attn),
            "use_var_attn": bool(args.use_var_attn),
            "temporal_fusion_mode": args.temporal_fusion_mode,
        },
        "training_config": {
            "seq_len": args.seq_len,
            "train_stride": args.train_stride or args.seq_len // 2,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "scheduler_factor": args.scheduler_factor,
            "scheduler_patience": args.scheduler_patience,
            "min_lr": args.min_lr,
            "train_missing_ratio": args.train_missing_ratio,
            "train_missing_type": args.train_missing_type,
            "val_missing_ratio": args.val_missing_ratio,
            "validation_missing_types": args.validation_missing_types,
            "validation_missing_ratios": args.validation_missing_ratios,
            "validation_worst_weight": args.validation_worst_weight,
            "use_augmentation": bool(args.use_augmentation),
            "aug_noise_std": args.aug_noise_std,
            "aug_scale_jitter": args.aug_scale_jitter,
            "aug_max_shift": args.aug_max_shift,
            "aug_dropout_rate": args.aug_dropout_rate,
            "aug_prob": args.aug_prob,
        },
        "loss_config": {
            "beta": args.beta,
            "missing_penalty": args.missing_penalty,
            "temporal_weight": args.temporal_weight,
            "wind_vector_weight": args.wind_vector_weight,
            "adaptive_temporal_scale": bool(args.adaptive_temporal_scale),
            "pointwise_loss": args.pointwise_loss,
            "feature_weights": args.feature_weights,
            "temporal_feature_weights": args.temporal_feature_weights,
        },
        "validation": validation_summary,
        "validation_conditions": validation_condition_summary,
        "innovation_scales": {
            "delta_scales": all_meta[0]["delta_scales"],
            "wind_delta_scale": all_meta[0]["wind_delta_scale"],
        },
        "test": test_summary,
        "log_path": log_filename,
    }
    summary_path = args.summary_path or os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.ablation_name}_{loss_tag}_summary.json",
    )
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎉 All runs completed! Report saved to {log_filename}")
    print(f"Summary saved to {summary_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Xinjiang')
    parser.add_argument('--data_root', type=str, default='./processed_data',
                        help='Directory containing <dataset>/obs and optional <dataset>/era5.')
    parser.add_argument('--train_missing_ratio', type=float, default=0.2)
    parser.add_argument('--train_missing_type', type=str, default='point_mcar')
    parser.add_argument('--val_missing_ratio', type=float, default=0.2)
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--train_stride', type=int, default=0, help='0 uses seq_len // 2.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Set to 0 to disable gradient clipping.')
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    parser.add_argument('--scheduler_patience', type=int, default=3)
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--use_era5', type=int, default=1, choices=[0, 1])
    parser.add_argument('--use_temporal_smooth', type=int, default=1, choices=[0, 1], help="1: use temporal block, 0: ablation without it")
    parser.add_argument('--use_time_attn', type=int, default=1)
    parser.add_argument('--use_var_attn', type=int, default=1)
    parser.add_argument(
        '--temporal_fusion_mode',
        choices=['independent', 'cascade', 'gated_cascade'],
        default='gated_cascade',
        help='Fusion inside the multi-scale temporal block.',
    )
    parser.add_argument('--ablation_name', type=str, default="FullModel", help="Name tag for the log file")
   
    # 新增的模型和Loss相关超参数
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=3)
    
    parser.add_argument('--beta', type=float, default=0.1, help="Beta for Smooth L1 loss")
    parser.add_argument('--missing_penalty', type=float, default=2.0, help="Penalty multiplier for missing parts in loss")
    parser.add_argument('--temporal_weight', type=float, default=0.2, help='Weight of the innovation-matching loss')
    parser.add_argument('--wind_vector_weight', type=float, default=0.05, help='Weight of the u/v vector-evolution loss')
    parser.add_argument('--adaptive_temporal_scale', type=int, default=1, choices=[0, 1], help='Normalize temporal losses with training-split innovation scales')
    parser.add_argument('--pointwise_loss', choices=['huber', 'mse'], default='huber', help='Pointwise reconstruction loss')
    parser.add_argument('--temporal_scale_floor', type=float, default=0.05, help='Minimum robust innovation scale')
    parser.add_argument('--selection_metric', choices=['mae', 'mse'], default='mae', help='Raw masked validation metric used for checkpoint selection')
    parser.add_argument(
        '--validation_missing_types', nargs='+', default=None,
        choices=['point_mcar', 'point_mar', 'seq_mcar', 'block_2d'],
        help='Optional validation suite. When provided, checkpoint selection uses all types and ratios.',
    )
    parser.add_argument(
        '--validation_missing_ratios', nargs='+', type=float, default=None,
        help='Ratios paired with every --validation_missing_types entry.',
    )
    parser.add_argument(
        '--validation_worst_weight', type=float, default=0.5,
        help='Selection score = mean validation metric + this weight times its worst condition.',
    )
    parser.add_argument('--selection_only', action='store_true', help='Skip full test evaluation after writing validation summary')
    parser.add_argument('--output_dir', type=str, default='results/loss_validation')
    parser.add_argument('--summary_path', type=str, default=None)
    parser.add_argument('--use_augmentation', type=int, default=0, choices=[0, 1])
    parser.add_argument('--aug_noise_std', type=float, default=0.01)
    parser.add_argument('--aug_scale_jitter', type=float, default=0.05)
    parser.add_argument('--aug_max_shift', type=int, default=1)
    parser.add_argument('--aug_dropout_rate', type=float, default=0.1)
    parser.add_argument('--aug_prob', type=float, default=0.5)
    parser.add_argument(
        '--feature_weights', nargs='+', type=float, default=[1.0] * 5,
        help='Independent weights for temperature, virtual temperature, u-wind, v-wind, pressure',
    )
    parser.add_argument(
        '--temporal_feature_weights', nargs='+', type=float, default=[1.0, 1.0, 0.5, 0.5, 1.0],
        help='Feature weights used only by the temporal innovation loss',
    )
    
    args = parser.parse_args()
    if len(args.feature_weights) != 5:
        parser.error('--feature_weights must provide exactly five values: temperature, virtual temperature, u-wind, v-wind, pressure.')
    if len(args.temporal_feature_weights) != 5:
        parser.error('--temporal_feature_weights must provide exactly five values: temperature, virtual temperature, u-wind, v-wind, pressure.')
    if args.weight_decay < 0 or args.grad_clip_norm < 0 or args.min_lr < 0:
        parser.error('--weight_decay, --grad_clip_norm and --min_lr must be non-negative.')
    if not 0 < args.scheduler_factor < 1:
        parser.error('--scheduler_factor must be in (0, 1).')
    if args.scheduler_patience < 0 or args.train_stride < 0:
        parser.error('--scheduler_patience and --train_stride must be non-negative.')
    if args.aug_noise_std < 0 or not 0 <= args.aug_scale_jitter < 1:
        parser.error('--aug_noise_std must be non-negative and --aug_scale_jitter must be in [0, 1).')
    if args.aug_max_shift < 0 or not 0 <= args.aug_dropout_rate < 1 or not 0 <= args.aug_prob <= 1:
        parser.error('Invalid augmentation parameters.')
    if (args.validation_missing_types is None) != (args.validation_missing_ratios is None):
        parser.error('--validation_missing_types and --validation_missing_ratios must be provided together.')
    if args.validation_missing_ratios is not None and any(not 0 < ratio < 1 for ratio in args.validation_missing_ratios):
        parser.error('--validation_missing_ratios must be in (0, 1).')
    if args.validation_worst_weight < 0:
        parser.error('--validation_worst_weight must be non-negative.')
    main(args)
