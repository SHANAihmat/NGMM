# NGMM: Noise-Aware Gated Multi-Scale Modulation

This directory is the reproducibility package for the paper's automatic
weather-station (AWS) imputation experiments. It contains the NGMM model,
mask-aware data loader, training/evaluation entry point, baseline adapters,
ablation runners, hyperparameter-search runners, and a preprocessing utility.
Raw data, private AWS observations, checkpoints, logs, and generated results
are intentionally excluded.

## What NGMM does

NGMM reconstructs missing multivariate hourly AWS observations. Its three
main components are:

1. **Gated Multi-Scale Temporal Refinement (GMTR).** Dilated temporal
   convolutions use fixed rates `(7, 3, 1)` to form coarse, medium, and fine
   receptive fields. Coarse features condition finer branches through learned
   sigmoid gates, while independent paths preserve local fluctuations.
2. **Factorized Temporal-Variable Attention (FTVA).** Temporal self-attention
   is applied to each station-variable sequence, followed by contemporaneous
   cross-variable attention. This preserves the distinction between temporal
   evolution and physical variable coupling.
3. **Domain-guided robust loss.** Huber reconstruction, MAD-normalized first
   differences, and joint zonal/meridional wind-vector innovation regularize
   values, temporal evolution, and wind consistency. Optional ERA5 fields are
   used as contextual backgrounds; the model predicts station-specific
   residual corrections instead of copying ERA5 values.

## Repository layout

```text
main.py                         Reproducible NGMM train/evaluate entry point
model.py                        NGMM architecture and loss functions
dataset.py                      Scalers, PyGrinder masks, DataLoaders
scripts/prepare_datasets.py     Raw-to-split preprocessing utility
scripts/run_ngmm_example.sh    Minimal training example
scripts/run_all_regions.sh      Four-region evaluation launcher
scripts/run_ablation.py         Component ablations
scripts/run_multiscale_fusion_ablation.py  GMTR fusion ablations
scripts/run_robust_hyperparameter_search.py Focused validation search
scripts/run_full_hyperparameter_sensitivity.py One-factor sensitivity search
scripts/run_case_study_and_downstream.py Qualitative/downstream experiments
baselines/                      Statistical, PyPOTS, diffusion, and pretrained adapters
visualization/                  Figure and table-generation scripts
configs/datasets.example.json   Timestamp/split configuration template
configs/ngmm_default.json       Paper-reported default NGMM configuration
requirements.txt                Pip dependencies
environment.yml                 Conda environment template
```

## Environment

The reported experiments used Python 3.9, PyTorch 2.5.1 with CUDA, NumPy,
PyGrinder, and PyPOTS. The exact CUDA runtime can vary with the installed
driver. A GPU is strongly recommended for neural baselines and large windows.

```bash
conda env create -f environment.yml
conda activate ngmm
# or: pip install -r requirements.txt
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
PY
```

## Data contract and privacy

The loader expects one directory per region under a configurable data root:

```text
processed_data/
  France/obs/{train,val,test}.npy
  France/era5/{train,val,test}.npy       # optional
```

Each array has shape `[stations, time, 5]`, with variables ordered as
`TMP, DEW, U_W, V_W, PRE`. AWS and ERA5 arrays must be temporally aligned.
ERA5 is spatially aligned to stations before entering this package, typically
with bilinear interpolation at station coordinates. The loader fits all
normalization statistics on the training split only.

The `France`, `Australia`, and `Zhejiang` AWS sources correspond to the public
Weather-5K data release. The Xinjiang/TC AWS observations are proprietary and
must not be redistributed. ERA5 itself is publicly available from the
Copernicus Climate Data Store. Reviewers can reproduce the public regions;
private-region results require access supplied by the authors.

To create split files from raw arrays:

```bash
python scripts/prepare_datasets.py \
  --raw-root /path/to/raw_arrays \
  --output-root ./processed_data \
  --config configs/datasets.example.json \
  --datasets France Australia Zhejiang
```

The raw directory should contain `obs.npy` and, when available, `era5.npy`.
The script only slices and writes splits; masking and normalization happen at
training/evaluation time.

## Quick start: NGMM

Run from the package root. `--data_root` may point anywhere containing the
prepared dataset directories.

```bash
python main.py \
  --dataset France --data_root ./processed_data --use_era5 0 \
  --hidden_dim 256 --num_layers 3 --num_heads 4 --kernel_size 3 \
  --seq_len 24 --train_stride 12 --lr 5e-4 --weight_decay 1e-4 \
  --beta 0.1 --missing_penalty 2.0 --temporal_weight 0.1 \
  --wind_vector_weight 0.025 --adaptive_temporal_scale 1 \
  --pointwise_loss huber --train_missing_type point_mcar \
  --train_missing_ratio 0.2 --val_missing_ratio 0.1 --batch_size 16 \
  --epochs 100 --num_runs 5 --output_dir ./results/france_ngmm
```

The same command with `--use_era5 1` uses `dataset/era5/{train,val,test}.npy`
when those files are present. If ERA5 is requested but unavailable, the code
prints a warning and records the fallback in the summary JSON.

For a shorter smoke test:

```bash
DATASET=France OUT_DIR=./results/smoke bash scripts/run_ngmm_example.sh
```

To verify the installation without any dataset:

```bash
python scripts/smoke_test.py
```

To run all four regions sequentially with their reported region-specific
settings:

```bash
DATA_ROOT=./processed_data USE_ERA5=0 RUNS=5 bash scripts/run_all_regions.sh
```

## Missingness and evaluation

The loader uses PyGrinder to generate deterministic validation/test masks and
resampled training masks. Supported mechanisms are `point_mcar`, `point_mar`,
`point_mnar_x`, `point_mnar_t`, `point_mnar_nonuniform`, `seq_mcar`, `seq`, and
`block_2d`. Metrics are computed only on withheld entries after inverse
transformation to physical units. The main paper reports MAE and RMSE; the
training summary also stores masked MSE, validation selection scores, and
condition-level metrics.

## Reproducing analyses

All runners are resumable and write JSON summaries plus text logs. Commands
should be launched from this directory:

```bash
python scripts/run_robust_hyperparameter_search.py --datasets France Australia Zhejiang Xinjiang
python scripts/run_full_hyperparameter_sensitivity.py \
  --datasets France Zhejiang \
  --parameters hidden_dim num_layers num_heads kernel_size seq_len lr weight_decay temporal_weight \
  --run-id reviewer_hpo_01
python scripts/run_ablation.py --num-runs 5
python scripts/run_multiscale_fusion_ablation.py --num-runs 5
```

The search scripts select configurations using validation data; the test split
is reserved for final reporting. If a job is interrupted, rerun the same
command and existing summary JSON files are skipped. The manifest records each
trial status and command line.

The scripts under `visualization/` regenerate paper figures/tables from the
JSON/CSV outputs produced by the experiment runners. They intentionally do not
ship those outputs, so pass each script the corresponding result directory
when running locally.

For the legacy main-comparison plotting script, set `NGMM_RESULTS_DIR`,
`NGMM_ERA5_RESULTS_DIR`, `BASELINE_RESULTS_DIR`, and `FIGURE_OUTPUT_DIR` to
your local paths before execution. The newer case-study and hyperparameter
plotters expose `--result-dir`/`--input` command-line arguments directly.

## Baseline adapters

`baselines/` includes Mean, Median, LOCF, SAITS, HELIX, TEFN, PatchTST,
ImputeFormer, CSDI, FSDI, Time-LLM, MOMENT, TimesNet, and TimeMixer++
adapters. Install optional dependencies only for the baselines being used.
Most `*_run_all.py` scripts expect to be launched from `baselines/` and find
data at `../processed_data`; adjust dataset/output arguments for another
layout. The FSDI external source is deliberately not bundled; follow its
upstream repository instructions before running that adapter.

## Reproducibility notes

- Record seeds, masks, split boundaries, and full command lines with every run.
- Normalization and robust innovation scales are fitted on training data only.
- NGMM shares parameters across stations but performs no explicit inter-station
  message passing.
- Select hyperparameters on validation data before generating final test scores.

## Citation and license

Please cite the accompanying paper and upstream Weather-5K, ERA5, PyGrinder,
PyPOTS, and baseline papers. No license is asserted for third-party baseline
implementations; preserve their upstream license notices. Add the authors'
chosen license before public release.
