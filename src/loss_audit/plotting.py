"""Static scientific figures; the underlying numbers are saved as JSON."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

from .lattice import ExponentialLattice


def make_plots(report: dict, output: Path) -> None:
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        _bounds_plot(report, output)
        _alias_plot(report, output)


def _bounds_plot(report: dict, output: Path) -> None:
    retention = 1000.0
    rows = [r for r in report["discretization"] if r["retention"] == retention]
    ref = next(r for r in report["continuous_reference"] if r["retention"] == retention)
    steps = np.array([r["step"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
    for ax, metric, title in zip(
        axes, ("exceedance", "stop_loss"), ("Exceedance probability P(S > 1,000)", "Stop-loss expectation E[(S - 1,000)+]")
    ):
        lo = [r["floor"][metric] for r in rows]
        hi = [r["ceil"][metric] for r in rows]
        ax.fill_between(steps, lo, hi, color="#cfdeed", alpha=0.8)
        ax.plot(steps, lo, "o-", color="#236192", label="Round each claim down")
        ax.plot(steps, hi, "o-", color="#c96929", label="Round each claim up")
        ax.axhline(ref[metric], color="#202a35", ls="--", label="Continuous Erlang-mixture reference")
        ax.set(xscale="log", title=title, xlabel="Claim-size grid step h (finer to the right)")
        ax.set_xticks(steps, [f"{s:g}" for s in steps])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.invert_xaxis()
        ax.grid(alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_ylabel("Synthetic loss units")
    fig.suptitle("Discretization is controlled by bracketing, not by normalizing", fontsize=13)
    fig.savefig(output / "discretization.png", dpi=160)
    plt.close(fig)


def _alias_plot(report: dict, output: Path) -> None:
    config = report["parameters"]
    model = ExponentialLattice(config["rate"], config["severity_mean"], config["aliasing_step"], "floor")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
    x = model.step * np.arange(128)
    axes[0].plot(x, model.aggregate_prefix(127), color="#202a35", lw=2, label="Unfolded Panjer prefix")
    for n, color, style in ((128, "#c96929", "-"), (512, "#236192", "--")):
        axes[0].plot(x, model.fft_periodic(n).probabilities[:128], style, color=color, label=f"Circular FFT, period {n * model.step:,.0f}")
    axes[0].set(title="Wrapped tail mass inflates low-loss cells", xlabel="Synthetic aggregate loss", ylabel="Probability per lattice cell")
    axes[0].legend(fontsize=8)
    rows = report["aliasing"]
    periods = [r["period"] for r in rows]
    # Display floor only; full signed floating-point diagnostics remain in JSON.
    axes[1].loglog(periods, [max(r["panjer_mass_outside_period"], 1e-15) for r in rows], "o-", color="#c96929", label="Mass outside FFT period (error bound)")
    axes[1].loglog(periods, [max(r["exceedance_understatement"], 1e-15) for r in rows], "s--", color="#236192", label="Tail probability understatement at 1,000")
    axes[1].axhline(config["absolute_alias_budget"], ls=":", color="#53665a", label="Absolute alias budget: 1e-6")
    axes[1].set(title="All four FFT distributions sum to one", xlabel="FFT period = cells × step", ylabel="Absolute probability error (display floor 1e-15)")
    axes[1].set_xticks(periods, [f"{p:,.0f}" for p in periods])
    axes[1].xaxis.set_minor_locator(NullLocator())
    axes[1].legend(fontsize=8, loc="best")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Circular aliasing is a different error from claim discretization", fontsize=13)
    fig.savefig(output / "aliasing.png", dpi=160)
    plt.close(fig)
