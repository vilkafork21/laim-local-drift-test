"""Порт all_results — строгий JSON: платформа и UI не принимают NaN."""

from __future__ import annotations

import json
import math

import main as drift


def _small_oos_result() -> dict:
    res = {
        "report": {"semaphore": "gray"},
        "precomputed": {
            "metric_value": math.nan,
            "metric_value_estimate": math.nan,
            "reliability": {
                "mean": math.nan, "median": math.nan, "q05": math.nan,
                "share_below_threshold": 1.0,
            },
        },
    }
    return drift.report_valtest_local_drift(res, drift._SEMAPHORE_TITLE["gray"])["all_results"]


def test_gray_small_oos_publishes_null_instead_of_nan():
    light = _small_oos_result()

    json.dumps(light, allow_nan=False)
    assert light["metric_value"] is None
    assert light["metric_value_estimate"] is None
    assert light["reliability_mean"] is None
    assert light["drop_estimate"] is None
    assert light["share_uncovered"] == 1.0
    assert light["status"] == "not_computable"
