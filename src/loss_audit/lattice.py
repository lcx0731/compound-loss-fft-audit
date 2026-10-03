"""Poisson Panjer recursion and an explicitly circular FFT calculation.

All severities and totals are nonnegative. A prefix is never renormalized:
missing probability represents real mass above its last support point.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np


def nonnegative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def integer(value: int, minimum: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def probability_prefix(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be a nonempty finite one-dimensional array")
    if np.any(p < 0) or p.sum() > 1 + 1e-12:
        raise ValueError("a probability prefix must be nonnegative and sum to at most 1")
    return p


def panjer_poisson(rate: float, severity_prefix: np.ndarray) -> np.ndarray:
    """Return aggregate probabilities through the same last lattice index.

    For severity p and aggregate g:
      g[0] = exp(rate * (p[0] - 1))
      g[k] = rate/k * sum(j * p[j] * g[k-j], j=1..k).
    Nonzero p[0] is allowed. The severity array may omit its upper tail.
    Runtime is O(K^2), memory O(K), for K output cells.
    """
    nonnegative(rate, "rate")
    p = probability_prefix(severity_prefix)
    g = np.zeros(p.size)
    g[0] = np.exp(rate * (p[0] - 1))
    if g[0] == 0:
        raise FloatingPointError("initial aggregate mass underflowed; reduce the frequency")
    weighted = np.arange(p.size) * p
    for k in range(1, p.size):
        g[k] = rate / k * np.dot(weighted[1 : k + 1], g[k - 1 :: -1])
    return g


@dataclass(frozen=True)
class Risk:
    exceedance: float
    stop_loss: float


def prefix_risk(aggregate_prefix: np.ndarray, step: float, mean: float) -> Risk:
    """Risk at d = step * last_index, using the model's *full* exact mean.

    The tail is P(S > d), not P(S >= d). Stop-loss is E[(S-d)+].
    The limited-loss identity retains tail expectation without cutting it off.
    This is not intended for probabilities close to machine precision.
    """
    p = probability_prefix(aggregate_prefix)
    positive(step, "step")
    nonnegative(mean, "mean")
    support = step * np.arange(p.size)
    if mean + 1e-10 < np.dot(support, p):
        raise ValueError("mean cannot be below the prefix's partial first moment")
    retention = support[-1]
    tail = float(1 - p.sum())
    excess = float(mean - np.dot(support, p) - retention * tail)
    if tail < -1e-12 or excess < -1e-10:
        raise FloatingPointError("invalid risk from the supplied mass and mean")
    return Risk(max(0.0, tail), max(0.0, excess))


@dataclass(frozen=True)
class FourierResult:
    # Distribution of the aggregate lattice index modulo n, NOT an unbounded PMF.
    probabilities: np.ndarray
    imaginary_residual: float


@dataclass(frozen=True)
class ExponentialLattice:
    rate: float
    scale: float
    step: float
    rounding: Literal["floor", "ceil"]

    def __post_init__(self) -> None:
        nonnegative(self.rate, "rate")
        positive(self.scale, "scale")
        positive(self.step, "step")
        if self.rounding not in ("floor", "ceil"):
            raise ValueError("rounding must be floor or ceil")
        if not np.isfinite(self.step / self.scale) or self.step / self.scale < 1e-10:
            raise ValueError("step / scale must be finite and at least 1e-10")
        if not np.isfinite(self.rate * (self.scale + self.step)):
            raise ValueError("model first moment overflows floating point")

    @property
    def q(self) -> float:
        return float(np.exp(-self.step / self.scale))

    @property
    def one_minus_q(self) -> float:
        return float(-np.expm1(-self.step / self.scale))

    @property
    def severity_mean(self) -> float:
        # Geometric support starts at zero for floor and at one for ceil.
        numerator = self.q if self.rounding == "floor" else 1.0
        return self.step * numerator / self.one_minus_q

    def severity_prefix(self, last_index: int) -> np.ndarray:
        integer(last_index, 0, "last_index")
        k = np.arange(last_index + 1)
        if self.rounding == "floor":
            return self.one_minus_q * np.exp(-self.step / self.scale * k)
        p = np.zeros(last_index + 1)
        p[1:] = self.one_minus_q * np.exp(-self.step / self.scale * k[:-1])
        return p

    def aggregate_prefix(self, last_index: int) -> np.ndarray:
        return panjer_poisson(self.rate, self.severity_prefix(last_index))

    def risk_at(self, retention: float) -> Risk:
        nonnegative(retention, "retention")
        ratio = retention / self.step
        if not np.isfinite(ratio) or not np.isclose(ratio, round(ratio), atol=1e-9, rtol=0):
            raise ValueError("retention must lie on the lattice")
        p = self.aggregate_prefix(int(round(ratio)))
        return prefix_risk(p, self.step, self.rate * self.severity_mean)

    def fft_periodic(self, n: int) -> FourierResult:
        """Invert the exact infinite-support severity PGF at n roots of unity.

        H_floor(z) = (1-q)/(1-q*z); H_ceil(z) = z*H_floor(z).
        The result is P(J mod n = k), where S_lattice = step * J.
        Using the exact PGF isolates circular aliasing from severity truncation.
        No clipping or renormalization is performed, even for tiny FFT roundoff.
        """
        integer(n, 2, "n")
        z = np.exp(-2j * np.pi * np.arange(n) / n)
        # Algebraically 1-q*z; this form avoids subtraction near z=1.
        h = self.one_minus_q / (self.one_minus_q + self.q * (1 - z))
        if self.rounding == "ceil":
            h = z * h
        transform = np.exp(self.rate * (h - 1))
        result = np.fft.ifft(transform)
        imaginary = float(np.max(np.abs(result.imag)))
        if imaginary > 1e-10 or result.real.min() < -1e-10:
            raise FloatingPointError("FFT inversion exceeded the roundoff tolerance")
        return FourierResult(result.real, imaginary)
