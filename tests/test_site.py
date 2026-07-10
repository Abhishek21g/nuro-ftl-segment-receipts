from __future__ import annotations

from pathlib import Path


def test_site_has_launch_artifacts() -> None:
    site = Path("site")
    assert (site / "index.html").exists()
    assert (site / "styles.css").exists()
    assert (site / "app.js").exists()
    assert (site / "charts.js").exists()
    assert (site / "data/scenarios.json").exists()
    assert (site / "data/regression_report.json").exists()
    assert (site / "data/summary.json").exists()
    assert (site / "data/doctor.json").exists()
    assert (site / "demo/regression_report.html").exists()
    assert (site / "demo/receipt.html").exists()
    assert (site / "demo/rules.from-scan.yaml").exists()


def test_scenario_bundle_complete() -> None:
    catalog = Path("site/data/scenarios.json")
    assert catalog.exists()
    import json

    data = json.loads(catalog.read_text())
    assert len(data["scenarios"]) >= 3
    for scenario in data["scenarios"]:
        prefix = Path("site") / scenario["data_prefix"]
        assert (prefix / "regression_report.json").exists()
        assert (prefix / "summary.json").exists()
        assert (prefix / "doctor.json").exists()
