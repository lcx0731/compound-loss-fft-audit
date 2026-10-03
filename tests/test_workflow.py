import copy
import json

import pytest

from loss_audit.experiment import build_report, save_report, verify_report


def test_report_is_deterministic_and_has_all_expected_comparisons():
    first = build_report()
    assert first == build_report()
    assert len(first["discretization"]) == 15
    assert len(first["aliasing"]) == 4
    assert len(first["validation"]) == 7
    assert all(first["validation"].values())


def test_artifacts_contain_the_verified_numbers_and_real_figures(tmp_path):
    report = build_report()
    save_report(report, tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text()) == report
    for name in ("discretization.png", "aliasing.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (tmp_path / name).stat().st_size > 10_000


def test_corrupt_results_fail_the_validation_gate():
    report = copy.deepcopy(build_report())
    report["discretization"][0]["floor"]["exceedance"] = 1
    with pytest.raises(ArithmeticError, match="bracketed"):
        verify_report(report)
