"""Непрерывные метки корзины: оценка на мониторинге публикуется в шкале меток."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_val.ann import ANN
from llm_val.sampler import AutoAsessorSampler
from llm_val.scorer import AutoAsessorScorer
from llm_val.utils import METRICS
from llm_val.valtest_local_drift_stability import valtest_local_drift_stability


class _OneHotEmbed:
    """Каждый уникальный вопрос — своя ось: копия запроса находит саму себя (sim = 1)."""

    def __init__(self, questions):
        self.axis = {q: i for i, q in enumerate(dict.fromkeys(questions))}

    def get_embedding(self, texts):
        vectors = np.zeros((len(texts), len(self.axis)), dtype=float)
        for row, text in enumerate(texts):
            vectors[row, self.axis[text]] = 1.0
        return vectors


def _estimate(low_label: float, high_label: float) -> float:
    questions = [f"вопрос {i}" for i in range(40)]
    targets = [low_label] * 10 + [high_label] * 30
    reference = pd.DataFrame({"question": questions, "answer": "", "target": targets})
    monitoring = pd.DataFrame({"question": questions[:10], "answer": "", "target": None})
    res = valtest_local_drift_stability(
        min_oos_samples=1, min_oot_samples=1,
        sampler=AutoAsessorSampler(agent_df=monitoring, real_df=reference),
        scorer=AutoAsessorScorer(metrics=METRICS),
        main_metric="target",
        model=_OneHotEmbed(questions),
        ann=ANN(),
        ann_config={"create_index": {"exact": True}, "search_query": {}},
        n_closest=1,
    )
    return res["precomputed"]["metric_value_estimate"]


def test_continuous_labels_estimate_is_in_label_scale():
    # Копии запросов с меткой 0.6 обязаны оцениваться в 0.6, а не в 0.0:
    # иначе снижение против среднего корзины ложно красное.
    assert _estimate(0.6, 1.0) == pytest.approx(0.6)


def test_binary_labels_keep_unit_scale():
    assert _estimate(0.0, 1.0) == pytest.approx(0.0)
