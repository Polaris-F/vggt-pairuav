#!/usr/bin/env python3
"""Aggregate per-seed ``pairuav.eval_val`` outputs for release custody."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "MAE_H_deg",
    "MAE_R_m",
    "endpoint_MAE_m",
    "SR@1m",
    "SR@2m",
    "SR@5m",
    "SR@10m",
    "angle_rel",
    "dist_rel",
    "final",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--system", default="range_C")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path, system: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if system not in data.get("results", {}):
        raise KeyError(f"{path} does not contain results.{system}")
    metrics = data["results"][system]
    missing = [name for name in METRICS if name not in metrics]
    if missing:
        raise KeyError(f"{path} is missing metrics: {missing}")
    return {
        "path": str(path.resolve()),
        "n_pairs": int(data["n_pairs"]),
        "metrics": {name: float(metrics[name]) for name in METRICS},
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in METRICS:
        values = [row["metrics"][name] for row in rows]
        out[name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return out


def main() -> None:
    args = parse_args()
    rows = [load_result(path, args.system) for path in args.results]
    if len({row["n_pairs"] for row in rows}) != 1:
        raise ValueError("per-seed evaluations use different pair counts")

    summary = {
        "system": args.system,
        "n_seeds": len(rows),
        "n_pairs": rows[0]["n_pairs"],
        "per_seed": rows,
        "aggregate": aggregate(rows),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# LaMP release evaluation",
        "",
        f"- System: `{args.system}`",
        f"- Seeds: `{len(rows)}`",
        f"- Pairs per seed: `{rows[0]['n_pairs']}`",
        "",
        "| metric | mean | sample std |",
        "|---|---:|---:|",
    ]
    for name in METRICS:
        value = summary["aggregate"][name]
        lines.append(f"| {name} | {value['mean']:.6f} | {value['std']:.6f} |")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
