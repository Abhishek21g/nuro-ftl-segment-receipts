from __future__ import annotations

from pathlib import Path


def test_site_has_launch_artifacts() -> None:
    site = Path("site")
    assert (site / "index.html").exists()
    assert (site / "data/regression_report.json").exists()
    assert (site / "data/summary.json").exists()
    assert (site / "data/doctor.json").exists()
    assert (site / "demo/regression_report.html").exists()
    assert (site / "demo/receipt.html").exists()
    assert (site / "demo/rules.from-scan.yaml").exists()
