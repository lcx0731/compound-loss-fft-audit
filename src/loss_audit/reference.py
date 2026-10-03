"""Independent continuous benchmark: a Poisson mixture of Erlang laws."""

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaincc
from scipy.stats import poisson

from .lattice import nonnegative, positive


@dataclass(frozen=True)
class ContinuousRisk:
    exceedance: float
    stop_loss: float
    omitted_probability_bound: float
    omitted_stop_loss_bound: float
    maximum_count: int


def continuous_risk(
    rate: float, scale: float, retention: float, tolerance: float = 1e-13
) -> ContinuousRisk:
    """Lower approximations with explicit count-truncation error bounds.

    Conditional on N=n>=1, S is Gamma(shape=n, scale=scale).
    Omitted probability <= P(N>K), and omitted stop-loss <=
    scale * E[N 1(N>K)] = scale * rate * P(N>=K).
    Bounds account for count truncation, not floating-point evaluation error.
    """
    nonnegative(rate, "rate")
    positive(scale, "scale")
    nonnegative(retention, "retention")
    if not np.isfinite(tolerance) or not 1e-15 <= tolerance < 0.1:
        raise ValueError("tolerance must lie in [1e-15, 0.1)")
    if not np.isfinite(rate * scale):
        raise ValueError("model first moment overflows floating point")
    if rate == 0:
        return ContinuousRisk(0.0, 0.0, 0.0, 0.0, 0)
    cutoff = int(poisson.isf(tolerance, rate))
    counts = np.arange(1, cutoff + 1)
    weights = poisson.pmf(counts, rate)
    ratio = retention / scale
    survival = gammaincc(counts, ratio)
    excess = scale * counts * gammaincc(counts + 1, ratio) - retention * survival
    return ContinuousRisk(
        exceedance=float(weights @ survival),
        stop_loss=float(weights @ excess),
        omitted_probability_bound=float(poisson.sf(cutoff, rate)),
        omitted_stop_loss_bound=float(scale * rate * poisson.sf(cutoff - 1, rate)),
        maximum_count=cutoff,
    )
