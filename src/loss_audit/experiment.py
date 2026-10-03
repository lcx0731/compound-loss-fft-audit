"""Run the deterministic audit, verify invariants, and write honest diagnostics."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .lattice import ExponentialLattice
from .reference import continuous_risk


def build_report() -> dict:
    rate, scale = 8.0, 100.0
    steps = [40.0, 20.0, 10.0, 5.0, 2.5]
    retentions = [1000.0, 1600.0, 2000.0]
    reference = [
        {"retention": d, **asdict(continuous_risk(rate, scale, d))} for d in retentions
    ]
    discretization = []
    for d in retentions:
        for step in steps:
            lower = ExponentialLattice(rate, scale, step, "floor").risk_at(d)
            upper = ExponentialLattice(rate, scale, step, "ceil").risk_at(d)
            discretization.append(
                {
                    "retention": d,
                    "step": step,
                    "floor": asdict(lower),
                    "ceil": asdict(upper),
                    "stop_loss_width": upper.stop_loss - lower.stop_loss,
                    "stop_loss_width_bound": rate * step,
                }
            )

    model = ExponentialLattice(rate, scale, 10.0, "floor")
    retention = 1000.0
    index = int(retention / model.step)
    true_tail = model.risk_at(retention).exceedance
    aliasing = []
    for n in (128, 256, 512, 1024):
        prefix = model.aggregate_prefix(n - 1)
        fft = model.fft_periodic(n)
        # Still sums to 1 even when substantial aggregate mass wraps around.
        circular_tail = float(1 - fft.probabilities[: index + 1].sum())
        missing = float(max(0.0, 1 - prefix.sum()))
        alias_l1 = float(np.abs(fft.probabilities - prefix).sum())
        aliasing.append(
            {
                "cells": n,
                "period": n * model.step,
                "fft_total_mass": float(fft.probabilities.sum()),
                "fft_min_cell": float(fft.probabilities.min()),
                "fft_imaginary_residual": fft.imaginary_residual,
                "panjer_mass_outside_period": missing,
                "alias_l1_vs_unfolded_prefix": alias_l1,
                "mass_identity_error": abs(alias_l1 - missing),
                "circular_exceedance": circular_tail,
                "exceedance_understatement": true_tail - circular_tail,
                "within_alias_budget": missing <= 1e-6,
            }
        )

    report = {
        "data": "Analytically specified synthetic model; no observed or private data.",
        "parameters": {
            "frequency": "Poisson",
            "rate": rate,
            "severity": "Exponential",
            "severity_mean": scale,
            "aggregate_mean": rate * scale,
            "aggregate_variance": 2 * rate * scale**2,
            "steps": steps,
            "retentions": retentions,
            "aliasing_step": model.step,
            "aliasing_rounding": "floor",
            "aliasing_retention": retention,
            "aliasing_reference_exceedance": true_tail,
            "absolute_alias_budget": 1e-6,
        },
        "continuous_reference": reference,
        "discretization": discretization,
        "aliasing": aliasing,
    }
    report["validation"] = verify_report(report)
    return report


def verify_report(report: dict) -> dict:
    """Fail before writing artifacts if the independent checks disagree."""
    checks = {}
    references = {x["retention"]: x for x in report["continuous_reference"]}
    brackets, widths, nested = [], [], []
    for row in report["discretization"]:
        ref = references[row["retention"]]
        for metric, error in (
            ("exceedance", ref["omitted_probability_bound"]),
            ("stop_loss", ref["omitted_stop_loss_bound"]),
        ):
            brackets.append(
                row["floor"][metric] <= ref[metric] + error + 1e-9
                and row["ceil"][metric] >= ref[metric] - 1e-9
            )
        widths.append(-1e-9 <= row["stop_loss_width"] <= row["stop_loss_width_bound"] + 1e-9)
    for d in references:
        rows = [r for r in report["discretization"] if r["retention"] == d]
        for coarse, fine in zip(rows, rows[1:]):
            for metric in ("exceedance", "stop_loss"):
                nested.append(
                    fine["floor"][metric] >= coarse["floor"][metric] - 1e-9
                    and fine["ceil"][metric] <= coarse["ceil"][metric] + 1e-9
                )
    checks["continuous_risk_bracketed"] = all(brackets)
    checks["stop_loss_width_within_rate_times_step"] = all(widths)
    checks["halving_step_nests_bounds"] = all(nested)
    checks["fft_mass_is_one_even_with_aliasing"] = all(
        abs(r["fft_total_mass"] - 1) < 1e-12 for r in report["aliasing"]
    )
    checks["alias_l1_equals_outside_mass"] = all(
        r["mass_identity_error"] < 1e-10 for r in report["aliasing"]
    )
    checks["exceedance_error_bounded_by_outside_mass"] = all(
        -1e-10 <= r["exceedance_understatement"] <= r["panjer_mass_outside_period"] + 1e-10
        for r in report["aliasing"]
    )
    checks["small_period_fails_and_large_period_passes_alias_budget"] = (
        not report["aliasing"][0]["within_alias_budget"]
        and report["aliasing"][-1]["within_alias_budget"]
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ArithmeticError(f"audit invariant failed: {', '.join(failed)}")
    return checks


def save_report(report: dict, output: Path) -> None:
    from .plotting import make_plots

    verify_report(report)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    make_plots(report, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = build_report()
    save_report(report, args.output)
    print(json.dumps({"output": str(args.output), "validation": report["validation"]}, indent=2))


if __name__ == "__main__":
    main()
