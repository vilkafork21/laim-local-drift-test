"""Reference UMR в формате тестового датасета: packed dialogue и flat с session_id."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from laim_monitoring import MonitoringContractError, unitize


def _contract(mode: str) -> dict:
    return {
        "contract_version": "laim-monitoring-metric.v2", "umr_version": "laim-umr.v2",
        "status": "computed", "basket_id": "CI1", "name": "quality", "score_column": "main_metric",
        "assessment_mode": mode,
        "scoring": {
            "method": "identity",
            "sources": [{
                "source_id": "source_1", "column_name": "score_metric", "role": "final_score",
                "normalization": "numeric", "polarity": "direct",
            }],
            "missing_policy": "fail", "majority_denominator": None,
        },
        "aggregation": {"method": "mean", "weight_column": None},
        "baseline": {
            "value": 0.5, "scale": "ratio", "value_source": "validation_report",
            "reported_value": 0.5, "reported_scale": "ratio", "recomputed_value": 0.5,
            "reconciliation": "match",
        },
        "primary_validation": {
            "threshold": None, "comparator": None, "scale": "ratio", "verdict": None,
            "affects_monitoring": False,
        },
        "evidence": {},
    }


def test_packed_dialogue_reference_is_unitized_per_session():
    frame = pd.DataFrame({
        "session_id": ["s1", "s2"],
        "dialogue": ["[('q1', 'hi', 'hello'), ('q2', 'bye', 'see you')]", "[('q3', 'x', 'y')]"],
        "input_query_count": [1, 1],
        "score_metric": [1.0, 0.0],
        "main_metric": [1.0, 0.0],
    })
    units = unitize(frame, _contract("dialogue"))
    assert len(units) == 2
    assert [turn["input_query"] for turn in units["dialogue"].iloc[0]] == ["hi", "bye"]
    assert units["source_1"].tolist() == [1.0, 0.0]
    assert units["main_metric"].tolist() == [1.0, 0.0]


def test_flat_reference_with_session_id_keeps_turn_history():
    frame = pd.DataFrame({
        "session_id": ["s1", "s1", "s2"],
        "query_id": ["q1", "q2", "q3"],
        "input_query_count": [1, 1, 1],
        "input_query": ["hi", "bye", "x"],
        "output_answer": ["hello", "see you", "y"],
        "score_metric": [1.0, 0.0, 1.0],
        "main_metric": [1.0, 0.0, 1.0],
    })
    units = unitize(frame, _contract("turn_with_history"))
    assert len(units) == 3
    assert [turn["input_query"] for turn in units["assessment_context"].iloc[1]["history"]] == ["hi"]


def test_flat_reference_without_canonical_columns_is_rejected():
    with pytest.raises(MonitoringContractError):
        unitize(pd.DataFrame({"question": ["q"], "answer": ["a"]}), _contract("qa"))


def test_drift_frames_from_packed_reference_and_packed_monitoring():
    """Формы выходов adapter (packed) и TDC (packed) согласуются в drift."""
    from laim_monitoring import prepare_drift_frames

    contract = _contract("dialogue")
    reference = pd.DataFrame({
        "session_id": ["s1", "s2"],
        "dialogue": [
            "[('t1', 'вопрос один', 'ответ один'), ('t2', 'вопрос два', 'ответ два')]",
            "[('t3', 'вопрос три', 'ответ три')]",
        ],
        "input_query_count": [1, 1],
        "score_metric": [1.0, 0.0],
        "main_metric": [1.0, 0.0],
    })
    monitoring = pd.DataFrame({
        "scenario": ["a", "b"],
        "session_id": ["m1", "m2"],
        "dialogue": [
            "[('mt1', 'наблюдённый вопрос', 'наблюдённый ответ')]",
            "[('mt2', 'ещё вопрос', 'ещё ответ'), ('mt3', 'и ещё', 'и ответ')]",
        ],
        "input_query_count": [1, 1],
    })

    ref_frame, mon_frame = prepare_drift_frames(reference, monitoring, contract)

    assert len(ref_frame) == 2  # единица drift — диалог
    assert len(mon_frame) == 2
    assert ref_frame["target"].tolist() == [1.0, 0.0]
    assert "вопрос один" in ref_frame["question"].iloc[0]


def test_drift_frames_from_flat_monitoring_with_session_id():
    """qa/turn_with_history: flat monitoring TDC без служебных колонок."""
    from laim_monitoring import prepare_drift_frames

    contract = _contract("turn_with_history")
    reference = pd.DataFrame({
        "session_id": ["s1", "s1"],
        "query_id": ["q1", "q2"],
        "input_query_count": [1, 1],
        "input_query": ["в1", "в2"],
        "output_answer": ["о1", "о2"],
        "score_metric": [1.0, 0.0],
        "main_metric": [1.0, 0.0],
    })
    monitoring = pd.DataFrame({
        "scenario": ["r", "r"],
        "session_id": ["m1", "m1"],
        "query_id": ["mq1", "mq2"],
        "input_query_count": [1, 1],
        "input_query": ["нв1", "нв2"],
        "output_answer": ["но1", "но2"],
    })

    ref_frame, mon_frame = prepare_drift_frames(reference, monitoring, contract)

    assert len(ref_frame) == 2
    assert len(mon_frame) == 2
    # История реплик сессии входит в question drift-фрейма
    assert "в1" in ref_frame["question"].iloc[1]


@pytest.mark.parametrize("mode", ["qa", "dialogue"])
def test_main_accepts_qa_and_dialogue(mode, monkeypatch):
    import main as drift

    if mode == "qa":
        reference = pd.DataFrame({
            "query_id": ["r1", "r2"],
            "input_query": ["вопрос 1", "вопрос 2"],
            "output_answer": ["ответ 1", "ответ 2"],
            "score_metric": [1.0, 0.0],
            "main_metric": [1.0, 0.0],
        })
        monitoring = reference.iloc[:1].drop(
            columns=["score_metric", "main_metric"]
        )
    else:
        reference = pd.DataFrame({
            "session_id": ["r1", "r2"],
            "dialogue": [
                repr([("r1-1", "вопрос 1", "ответ 1"), ("r1-2", "уточнение", "ответ")]),
                repr([("r2-1", "вопрос 2", "ответ 2")]),
            ],
            "input_query_count": [1, 1],
            "score_metric": [1.0, 0.0],
            "main_metric": [1.0, 0.0],
        })
        monitoring = reference.iloc[:1].drop(
            columns=["score_metric", "main_metric"]
        )

    captured = {}

    def fake_valtest(*, sampler, **_kwargs):
        captured["sizes"] = (len(sampler.train["X"]), len(sampler.test["X"]))
        return {
            "report": {"semaphore": "green"},
            "precomputed": {
                "metric_value": 0.5,
                "metric_value_estimate": 0.5,
                "reliability": {
                    "mean": 1.0,
                    "median": 1.0,
                    "q05": 1.0,
                    "share_below_threshold": 0.0,
                },
            },
        }

    monkeypatch.setattr(
        drift, "Config", lambda: SimpleNamespace(contour_configs={})
    )
    monkeypatch.setattr(drift, "GigaEmbed", lambda **_kwargs: object())
    monkeypatch.setattr(drift, "ANN", lambda: object())
    monkeypatch.setattr(drift, "valtest_local_drift_stability", fake_valtest)

    result = drift.main(reference, monitoring, _contract(mode))

    assert captured["sizes"] == (2, 1)
    assert result["all_results"]["test_name"] == "local_drift"
    assert result["all_results"]["status"] == "computed"
    assert result["all_results"]["metric_value"] == 0.5
    assert result["all_results"]["metric_value_estimate"] == 0.5


def test_not_computable_metric_skips_drift_computation(monkeypatch):
    import main as drift

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Вычислительный путь не должен запускаться")

    monkeypatch.setattr(drift, "prepare_drift_frames", forbidden)
    monkeypatch.setattr(drift, "Config", forbidden)
    monkeypatch.setattr(drift, "GigaEmbed", forbidden)
    monkeypatch.setattr(drift, "valtest_local_drift_stability", forbidden)

    result = drift.main(
        object(),
        object(),
        {
            "contract_version": "laim-monitoring-metric.v2",
            "umr_version": "laim-umr.v2",
            "status": "not_computable",
            "reason_code": "ambiguous_baseline",
            "reason": "baseline нельзя определить однозначно",
        },
    )

    light = result["all_results"]
    assert light["color"] == "gray"
    assert light["status"] == "not_computable"
    assert light["reason_code"] == "ambiguous_baseline"
    assert light["reason"] == "baseline нельзя определить однозначно"
    assert light["test_name"] == "local_drift"


def test_descriptor_defaults_match_runtime_contract():
    root = Path(__file__).resolve().parents[1]
    descriptor = json.loads((root / "descriptor.json").read_text())
    components = descriptor["ui"]["settings"][0]["components"][0]["config"][
        "components"
    ]
    defaults = {component["parameter"]: component.get("defaultValue") for component in components}
    assert "red_threshold" not in defaults and "green_threshold" not in defaults
    assert defaults["reliability_threshold"] == 0.2
    assert defaults["uncovered_amber_share"] == 0.3
    assert defaults["uncovered_red_share"] == 0.5
    assert defaults["min_reference_units"] == 30
    assert defaults["min_monitoring_units"] == 30
    monitoring_port = next(
        port for port in descriptor["ports"] if port["name"] == "monitoring_umr"
    )
    assert "monitoring_umr" in monitoring_port["description"]
    assert "parquet_test_dataset" not in monitoring_port["description"]
    run_config = descriptor["script"]["runConfiguration"]
    assert run_config["libraryDependencies"] == ["requirements.txt"]
    assert all((root / path).is_file() for path in run_config["sourceFiles"])


def test_yellow_semaphore_is_published_as_amber_with_local_drift_title():
    # Внутренний yellow должен уходить платформе как amber, а заголовок —
    # описывать локальный дрифт, а не динамику КМ (LAIM-0004, LAIM-0045).
    import main as drift

    res = {"report": {"semaphore": "yellow"},
           "precomputed": {"metric_value": 0.9, "metric_value_estimate": 0.7,
                           "reliability": {"mean": 1.0, "share_below_threshold": 0.0}}}
    result = drift.report_valtest_local_drift(res, drift._SEMAPHORE_TITLE["yellow"])
    light = result["all_results"]
    assert light["color"] == "amber"
    assert light["calculated_traffic_lights"]["test_light"] == "amber"
    assert "покрытие" in light["calculated_traffic_lights"]["semaphore_title"].lower()
    assert "динамики ключевой метрики" not in light["calculated_traffic_lights"]["semaphore_title"]
    assert light["informative"] is True


def test_gray_verdict_explains_itself():
    # Серый светофор без причины неотличим от сбоя: причина обязана быть словами.
    import math

    import main as drift

    res = {"report": {"semaphore": "gray"},
           "precomputed": {"metric_value": 0.9, "metric_value_estimate": math.nan,
                           "reliability": {"mean": math.nan, "share_below_threshold": 1.0}}}
    light = drift.report_valtest_local_drift(res, drift._SEMAPHORE_TITLE["gray"])["all_results"]
    assert light["status"] == "not_computable"
    assert "OOS" in light["reason"]


def test_coverage_colour_depends_only_on_uncovered_share():
    # Карточка 6.3.7: цвет — технический сигнал по доле непокрытых запросов;
    # уровень метрики корзины и оценка по соседям цвет не меняют.
    from llm_val.valtest_local_drift_stability import report_valtest_local_drift_stability

    def colour(share, mean=0.9, estimate=0.2, metric=0.95):
        stats = {"mean": mean, "median": mean, "q05": mean, "share_below_threshold": share}
        return report_valtest_local_drift_stability(
            {"target": metric}, estimate, stats, "target",
            reliability_threshold=0.2, uncovered_amber=0.3, uncovered_red=0.5,
        )["semaphore"]

    assert colour(0.0) == "green"          # оценка по соседям 0.2 при корзине 0.95 — не красный
    assert colour(0.0, metric=0.3) == "green"  # низкий уровень корзины — не жёлтый
    assert colour(0.31) == "yellow"
    assert colour(0.0, mean=0.1) == "yellow"
    assert colour(0.51) == "red"


def test_small_monitoring_sample_is_not_assessed_with_reason(monkeypatch):
    import math

    import main as drift

    res = {"report": {"semaphore": "gray"},
           "precomputed": {"metric_value": 0.9, "metric_value_estimate": math.nan,
                           "reliability": {"mean": math.nan, "share_below_threshold": 1.0},
                           "reason_code": "insufficient_monitoring_units",
                           "reason": "OOT 3 единиц меньше минимума 30",
                           "n_oos": 120, "n_oot": 3, "n_closest": None}}
    light = drift.report_valtest_local_drift(res, drift._SEMAPHORE_TITLE["gray"])["all_results"]
    assert light["status"] == "not_computable"
    assert light["reason_code"] == "insufficient_monitoring_units"
    assert light["reason"] == "OOT 3 единиц меньше минимума 30"
    assert light["n_oos"] == 120 and light["n_oot"] == 3
