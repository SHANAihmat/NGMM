#!/usr/bin/env python3
"""Build the point-MCAR (10%) main-comparison table from experiment logs.

The script intentionally reads only completed, timestamped ``MultiTest``
baseline reports and the completed NGMM gated-cascade summaries.  It exports
the numerical records together with their source paths, so the paper table
can be audited and regenerated without manual transcription.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
TARGET_CONDITION = "point_mcar_ratio0.1"
DATASETS = ("France", "Australia", "Zhejiang", "Xinjiang")

# The display labels follow the climate-regime nomenclature used in the paper.
DATASET_LABELS = {
    "France": "France (MN)",
    "Australia": "Australia (MS)",
    "Zhejiang": "Zhejiang (SM)",
    "Xinjiang": "Xinjiang (TC)",
}

BASELINES = (
    ("Mean", "Mean"),
    ("Median", "Median"),
    ("LOCF", "LOCF"),
    ("SAITS", "SAITS"),
    ("CSDI", "CSDI"),
    ("TimesNet", "TimesNet"),
    ("TimeLLM", "Time-LLM"),
    ("MOMENT", "MOMENT"),
    ("ImputeFormer", "ImputeFormer"),
    ("TimeMixerPP", "TimeMixer++"),
    ("PatchTST", "PatchTST"),
    ("TEFN", "TEFN"),
    ("HELIX", "HELIX"),
    ("FSDI", "FSDI"),
)

# This is the proposed, consistent architecture used in all four regions.  We
# deliberately do not pick a different ablation branch per test metric.
NGMM_SUMMARIES = {
    dataset: ROOT
    / "results"
    / "multiscale_fusion_ablation"
    / dataset
    / "gated_cascade"
    / "summary.json"
    for dataset in DATASETS
}

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
CONDITION_RE = re.compile(
    r"^\s*>>>\s*(?:Test\s+Condition\s*:\s*)?(?P<condition>\S+)",
    flags=re.IGNORECASE | re.MULTILINE,
)
METRIC_RE = re.compile(
    rf"^\s*(?P<metric>mae|rmse)\s*:\s*(?P<mean>{NUMBER})"
    rf"\s*(?:±|\+/-|Â±)\s*(?P<std>{NUMBER})\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
RUNS_RE = re.compile(r"Number of Runs\s*:\s*(?P<runs>\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Metric:
    mean: float
    std: float


@dataclass(frozen=True)
class Result:
    method: str
    dataset: str
    mae: Metric
    rmse: Metric
    source: str
    num_runs: int | None


def latest_report(method_key: str, dataset: str) -> Path:
    """Return the newest report for one baseline/dataset pair.

    FSDI reports live in a timestamped subdirectory, hence recursive search.
    Text reports are preferred because all legacy baselines use that format.
    """
    result_root = ROOT / "baselines" / "results"
    candidates = list(result_root.rglob(f"{method_key}_{dataset}_MultiTest_*.txt"))
    if not candidates:
        raise FileNotFoundError(
            f"No MultiTest report found for {method_key} on {dataset} under {result_root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def section_for_condition(text: str, condition: str) -> str:
    matches = list(CONDITION_RE.finditer(text))
    for index, match in enumerate(matches):
        if match.group("condition").lower() == condition.lower():
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            return text[match.end() : end]
    raise ValueError(f"Condition {condition!r} was not found")


def parse_text_report(path: Path, method: str, dataset: str) -> Result:
    text = path.read_text(encoding="utf-8", errors="replace")
    section = section_for_condition(text, TARGET_CONDITION)
    metrics: dict[str, Metric] = {}
    for match in METRIC_RE.finditer(section):
        metrics[match.group("metric").lower()] = Metric(
            float(match.group("mean")), float(match.group("std"))
        )
    missing = {"mae", "rmse"}.difference(metrics)
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)} for {TARGET_CONDITION}")
    runs_match = RUNS_RE.search(text)
    return Result(
        method=method,
        dataset=dataset,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        source=str(path.relative_to(ROOT)),
        num_runs=int(runs_match.group("runs")) if runs_match else None,
    )


def parse_ngmm_summary(path: Path, dataset: str) -> Result:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["test"][TARGET_CONDITION]
    return Result(
        method="NGMM (w/o ERA5)",
        dataset=dataset,
        mae=Metric(**metrics["mae"]),
        rmse=Metric(**metrics["rmse"]),
        source=str(path.relative_to(ROOT)),
        num_runs=payload.get("num_runs"),
    )


def collect_results() -> list[Result]:
    results: list[Result] = []
    for method_key, display_name in BASELINES:
        for dataset in DATASETS:
            results.append(
                parse_text_report(latest_report(method_key, dataset), display_name, dataset)
            )
    for dataset, summary_path in NGMM_SUMMARIES.items():
        if not summary_path.is_file():
            raise FileNotFoundError(f"NGMM summary is missing: {summary_path}")
        results.append(parse_ngmm_summary(summary_path, dataset))
    return results


def is_metric_pair_valid(result: Result) -> bool:
    """Check the non-negotiable MAE <= RMSE identity for one evaluation set.

    Both values are computed over the same target positions.  A violation
    indicates that the source report mixes incompatible aggregation or scaling
    procedures, so it must not participate in paper-table ranking.
    """
    return (
        result.mae.mean >= 0.0
        and result.rmse.mean >= 0.0
        and result.mae.mean <= result.rmse.mean + 1e-12
    )


def ranks(results: Iterable[Result]) -> dict[tuple[str, str], dict[str, int]]:
    """Rank mean errors within each dataset/metric; lower is better."""
    result_list = list(results)
    output: dict[tuple[str, str], dict[str, int]] = {}
    for dataset in DATASETS:
        for metric_name in ("mae", "rmse"):
            ordered = sorted(
                (
                    result
                    for result in result_list
                    if result.dataset == dataset and is_metric_pair_valid(result)
                ),
                key=lambda result: getattr(result, metric_name).mean,
            )
            for rank, result in enumerate(ordered, start=1):
                output.setdefault((dataset, metric_name), {})[result.method] = rank
    return output


def metric_tex(metric: Metric, rank: int) -> str:
    text = f"{metric.mean:.4f} $\\pm$ {metric.std:.4f}"
    if rank == 1:
        return f"\\textbf{{{text}}}"
    if rank == 2:
        return f"\\underline{{{text}}}"
    return text


def make_latex(results: list[Result]) -> str:
    lookup = {(result.method, result.dataset): result for result in results}
    rank_lookup = ranks(results)
    methods = [name for _, name in BASELINES] + ["NGMM (w/o ERA5)"]
    lines = [
        "% Auto-generated by generate_main_comparison_table.py. Do not edit manually.",
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{Main comparison under point-wise MCAR missingness with a 10\\% missing ratio. Values are mean $\\pm$ standard deviation over repeated runs; lower is better. The best and second-best valid results in each column are highlighted in bold and underlined, respectively. $^\\dagger$The latest CSDI logs violate the necessary $\\mathrm{MAE}\\leq\\mathrm{RMSE}$ identity and are therefore excluded pending metric-pipeline correction and rerunning.}",
        "    \\label{tab:main_comparison_point_mcar_10}",
        "    \\scriptsize",
        "    \\setlength{\\tabcolsep}{2.7pt}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{lcccccccc}",
        "        \\toprule",
        "        \\multirow{2}{*}{Method} & \\multicolumn{2}{c}{France (MN)} & \\multicolumn{2}{c}{Australia (MS)} & \\multicolumn{2}{c}{Zhejiang (SM)} & \\multicolumn{2}{c}{Xinjiang (TC)} \\\\",
        "        \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}",
        "        & MAE & RMSE & MAE & RMSE & MAE & RMSE & MAE & RMSE \\\\",
        "        \\midrule",
    ]
    for method in methods:
        cells = [method]
        for dataset in DATASETS:
            result = lookup[(method, dataset)]
            if not is_metric_pair_valid(result):
                cells.extend(["--$^\\dagger$", "--$^\\dagger$"])
                continue
            cells.append(metric_tex(result.mae, rank_lookup[(dataset, "mae")][method]))
            cells.append(metric_tex(result.rmse, rank_lookup[(dataset, "rmse")][method]))
        lines.append("        " + " & ".join(cells) + " \\\\")
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}%",
            "    }",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(results: list[Result], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method",
                "dataset",
                "mae_mean",
                "mae_std",
                "rmse_mean",
                "rmse_std",
                "num_runs",
                "metric_pair_valid",
                "source",
            ),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "method": result.method,
                    "dataset": result.dataset,
                    "mae_mean": result.mae.mean,
                    "mae_std": result.mae.std,
                    "rmse_mean": result.rmse.mean,
                    "rmse_std": result.rmse.std,
                    "num_runs": result.num_runs,
                    "metric_pair_valid": is_metric_pair_valid(result),
                    "source": result.source,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "main_comparison_point_mcar_10",
        help="Directory for the generated .tex, .csv, and .json artifacts.",
    )
    args = parser.parse_args()

    results = collect_results()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = args.output_dir / "main_comparison_point_mcar_10.tex"
    csv_path = args.output_dir / "main_comparison_point_mcar_10.csv"
    json_path = args.output_dir / "main_comparison_point_mcar_10.json"
    tex_path.write_text(make_latex(results), encoding="utf-8")
    write_csv(results, csv_path)
    json_path.write_text(
        json.dumps(
            [
                {**asdict(result), "metric_pair_valid": is_metric_pair_valid(result)}
                for result in results
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {tex_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
