import numpy as np
import pytest

from loss_audit.lattice import ExponentialLattice
from loss_audit.reference import continuous_risk


def test_continuous_reference_at_zero_matches_known_mean_and_zero_atom():
    risk = continuous_risk(8, 100, 0)
    assert abs(risk.exceedance - (1 - np.exp(-8))) <= risk.omitted_probability_bound + 1e-14
    assert abs(risk.stop_loss - 800) <= risk.omitted_stop_loss_bound + 1e-11


def test_zero_rate_reference():
    risk = continuous_risk(0, 100, 1000)
    assert risk.exceedance == risk.stop_loss == risk.omitted_probability_bound == 0


def test_count_truncation_bounds_cover_a_tighter_reference():
    loose = continuous_risk(8, 100, 1000, tolerance=0.001)
    tight = continuous_risk(8, 100, 1000)
    assert 0 <= tight.exceedance - loose.exceedance <= loose.omitted_probability_bound
    assert 0 <= tight.stop_loss - loose.stop_loss <= loose.omitted_stop_loss_bound


def test_reference_scales_in_loss_units():
    original = continuous_risk(8, 100, 1000)
    scaled = continuous_risk(8, 1, 10)
    assert original.exceedance == pytest.approx(scaled.exceedance, abs=1e-14)
    assert original.stop_loss == pytest.approx(100 * scaled.stop_loss, abs=1e-10)


def test_reference_agrees_with_independent_seeded_simulation():
    rng = np.random.default_rng(20261003)
    counts = rng.poisson(8, size=200_000)
    totals = rng.gamma(counts, scale=100)
    risk = continuous_risk(8, 100, 1000)
    indicators = totals > 1000
    excess = np.maximum(totals - 1000, 0)
    assert abs(indicators.mean() - risk.exceedance) < 6 * indicators.std(ddof=1) / np.sqrt(totals.size)
    assert abs(excess.mean() - risk.stop_loss) < 6 * excess.std(ddof=1) / np.sqrt(totals.size)


@pytest.mark.parametrize("rounding", ["floor", "ceil"])
def test_fft_matches_explicit_modular_folding_of_panjer(rounding):
    model = ExponentialLattice(1.4, 2, 0.5, rounding)
    long_prefix = model.aggregate_prefix(1023)
    assert long_prefix.sum() == pytest.approx(1, abs=1e-13)
    folded = long_prefix.reshape(-1, 32).sum(axis=0)
    result = model.fft_periodic(32)
    np.testing.assert_allclose(result.probabilities, folded, atol=5e-15)
    assert result.imaginary_residual < 1e-14


def test_small_fft_period_passes_mass_check_but_fails_tail_accuracy():
    model = ExponentialLattice(8, 100, 10, "floor")
    circular = model.fft_periodic(128).probabilities
    assert circular.sum() == pytest.approx(1, abs=1e-14)
    naive_tail = 1 - circular[:101].sum()
    true_tail = model.risk_at(1000).exceedance
    assert true_tail - naive_tail > 0.05
    prefix = model.aggregate_prefix(127)
    np.testing.assert_allclose(np.abs(circular - prefix).sum(), 1 - prefix.sum(), atol=1e-14)


def test_large_fft_period_recovers_unfolded_cells_without_normalizing():
    model = ExponentialLattice(8, 100, 10, "floor")
    np.testing.assert_allclose(model.fft_periodic(1024).probabilities[:101], model.aggregate_prefix(100), atol=1e-14)


@pytest.mark.parametrize("n", [0, 1, 2.5, True])
def test_invalid_fft_sizes_are_rejected(n):
    with pytest.raises(ValueError):
        ExponentialLattice(8, 100, 10, "ceil").fft_periodic(n)


@pytest.mark.parametrize("retention", [0, 1000, 1600, 2000])
def test_lattice_bounds_bracket_reference_and_nest_on_refinement(retention):
    reference = continuous_risk(8, 100, retention)
    previous = None
    for h in [40, 20, 10, 5, 2.5]:
        lo = ExponentialLattice(8, 100, h, "floor").risk_at(retention)
        hi = ExponentialLattice(8, 100, h, "ceil").risk_at(retention)
        assert lo.exceedance <= reference.exceedance + 1e-11 <= hi.exceedance + 1e-10
        assert lo.stop_loss <= reference.stop_loss + 1e-8 <= hi.stop_loss + 1e-7
        assert hi.stop_loss - lo.stop_loss <= 8 * h + 1e-8
        if previous:
            assert lo.exceedance >= previous[0].exceedance - 1e-12
            assert hi.exceedance <= previous[1].exceedance + 1e-12
            assert lo.stop_loss >= previous[0].stop_loss - 1e-10
            assert hi.stop_loss <= previous[1].stop_loss + 1e-10
        previous = lo, hi
