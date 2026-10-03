import numpy as np
import pytest
from scipy.stats import poisson

from loss_audit.lattice import ExponentialLattice, panjer_poisson, prefix_risk


def test_unit_claims_recover_poisson_distribution():
    p = np.zeros(31)
    p[1] = 1
    np.testing.assert_allclose(panjer_poisson(4.5, p), poisson.pmf(np.arange(31), 4.5), atol=2e-16)


def test_zero_size_claims_thin_the_poisson_count():
    p = np.zeros(31)
    p[:2] = [0.7, 0.3]
    np.testing.assert_allclose(panjer_poisson(4.5, p), poisson.pmf(np.arange(31), 4.5 * 0.3), atol=2e-16)


def test_panjer_matches_independent_direct_convolutions():
    severity = np.zeros(41)
    severity[:3] = [0.2, 0.3, 0.5]
    term = np.zeros(41)
    term[0] = 1
    expected = np.zeros(41)
    for count in range(41):
        expected += poisson.pmf(count, 2.3) * term
        term = np.convolve(term, severity)[:41]
    np.testing.assert_allclose(panjer_poisson(2.3, severity), expected, atol=5e-16)


def test_truncating_a_prefix_does_not_change_earlier_cells():
    model = ExponentialLattice(8, 100, 10, "floor")
    small = model.aggregate_prefix(20)
    large = model.aggregate_prefix(300)
    np.testing.assert_array_equal(small, large[:21])
    assert small.sum() < 0.1  # Omitted mass is retained as an omission, not normalized away.


@pytest.mark.parametrize("rounding", ["floor", "ceil"])
def test_geometric_severity_mass_and_mean(rounding):
    model = ExponentialLattice(8, 100, 10, rounding)
    p = model.severity_prefix(1000)
    assert p.sum() == pytest.approx(1, abs=1e-14)
    assert np.dot(np.arange(p.size) * model.step, p) == pytest.approx(model.severity_mean, abs=1e-10)


@pytest.mark.parametrize("rounding", ["floor", "ceil"])
def test_zero_frequency_is_a_point_mass_at_zero(rounding):
    model = ExponentialLattice(0, 100, 10, rounding)
    np.testing.assert_array_equal(model.aggregate_prefix(5), [1, 0, 0, 0, 0, 0])
    assert model.risk_at(100).exceedance == 0
    assert model.risk_at(100).stop_loss == 0


def test_risk_uses_strict_exceedance_and_full_mean():
    # P(S=0,1,2)=(.2,.3,.5), but only cells 0,1 are supplied.
    risk = prefix_risk(np.array([0.2, 0.3]), step=1, mean=1.3)
    assert risk.exceedance == pytest.approx(0.5)
    assert risk.stop_loss == pytest.approx(0.5)


@pytest.mark.parametrize("p", [[], [[1.0]], [-0.1, 1.1], [0.6, 0.6], [np.nan], [np.inf]])
def test_invalid_probability_prefix_is_rejected(p):
    with pytest.raises(ValueError):
        panjer_poisson(1, p)


@pytest.mark.parametrize("rate", [-1, np.inf, np.nan])
def test_invalid_rate_is_rejected(rate):
    with pytest.raises(ValueError):
        panjer_poisson(rate, [0, 1])


def test_panjer_initial_underflow_is_not_silently_reported_as_zero_risk():
    with pytest.raises(FloatingPointError, match="underflow"):
        panjer_poisson(1000, [0, 1])


@pytest.mark.parametrize("kwargs", [{"scale": 0}, {"step": -1}, {"rounding": "nearest"}, {"step": 1e-12}])
def test_invalid_lattice_parameters_are_rejected(kwargs):
    params = dict(rate=8, scale=100, step=10, rounding="floor")
    params.update(kwargs)
    with pytest.raises(ValueError):
        ExponentialLattice(**params)


def test_non_lattice_retention_is_rejected():
    with pytest.raises(ValueError, match="lattice"):
        ExponentialLattice(8, 100, 10, "floor").risk_at(101)
