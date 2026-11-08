from src.evaluate import check_thresholds

THRESHOLDS = {"accuracy": 0.80, "recall": 0.20, "f1": 0.25}


def test_model_passes_when_all_metrics_meet_thresholds():
    ok, _ = check_thresholds({"accuracy": 0.9, "recall": 0.3, "f1": 0.3}, THRESHOLDS)
    assert ok


def test_model_fails_when_one_metric_below_threshold():
    ok, rows = check_thresholds({"accuracy": 0.9, "recall": 0.1, "f1": 0.3}, THRESHOLDS)
    assert not ok
    assert [r["metric"] for r in rows if not r["pass"]] == ["recall"]


def test_missing_or_nan_metric_counts_as_failure():
    ok, _ = check_thresholds({"accuracy": 0.9, "f1": float("nan")}, THRESHOLDS)
    assert not ok
