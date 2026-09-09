"""
Тест на локальный дрифт запросов — точка входа.
"""

import logging
import math
from ast import literal_eval
from typing import Any

import pandas as pd

from config import Config
from giga_wraper import GigaEmbed
from llm_val.ann import ANN
from llm_val.sampler import AutoAsessorSampler
from llm_val.scorer import AutoAsessorScorer
from llm_val.utils import METRICS  # P3-5
from llm_val.valtest_local_drift_stability import (
    MIN_OOS_SAMPLES,
    valtest_local_drift_stability,
)
from laim_monitoring import prepare_drift_frames

from html_report import format_report_number, render_test_report


# =============================================================================
# ФУНКЦИИ ФОРМИРОВАНИЯ ОТЧЕТОВ
# =============================================================================




def html_report_valtest_local_drift(res: dict, semaphore_title: str) -> str:
    pre = res["precomputed"]
    reliability = pre.get("reliability", {})
    baseline, current = pre.get("metric_value"), pre.get("metric_value_estimate")
    delta = baseline - current if baseline is not None and current is not None else None
    rows = [
        ("Значение КМ на эталонной корзине (OOS)", format_report_number(baseline)),
        ("Оценка КМ на мониторинге (OOT)", format_report_number(current)),
        ("Абсолютное снижение D", format_report_number(delta)),
        ("Надёжность: среднее / медиана / 5-й перцентиль близости",
         " / ".join(format_report_number(reliability.get(key)) for key in ("mean", "median", "q05"))),
        (f"Доля непокрытых запросов (близость < {format_report_number(pre.get('reliability_threshold', 0.7), 2)})",
         format_report_number(reliability.get("share_below_threshold"), 1, percent=True)),
        ("Число ближайших соседей N", format_report_number(pre.get("n_closest"), 0)),
    ]
    return render_test_report(
        "6.3.6", "Локальный дрифт запросов",
        "Оценить ожидаемое качество ответов по семантически ближайшим запросам эталонной корзины "
        "и выявить долю запросов текущего потока, не покрытых эталоном.",
        rows, res["report"]["semaphore"],
        "Положительное снижение означает, что ожидаемая оценка на мониторинге ниже эталонной; "
        "отрицательное — выше. Высокая доля непокрытых запросов указывает на новые тематики "
        "и помогает выбрать запросы для дополнительной разметки. Оценка основана на сходстве "
        "запросов и не использует ответы решения или их оценки Автоасессором за период. Серый результат означает, "
        "что вывод о дрифте не получен.",
        f"СЗ выше E; не менее {MIN_OOS_SAMPLES} объектов на эталоне. Метки бинарные "
        "или непрерывные в известной шкале [0; 1]. Точный поиск по запросам, усечённым до 1000 символов.",
        "Пороги по умолчанию: зелёный — снижение D < 0,5 и зелёная КМ на эталоне; "
        "красный — D ≥ 0,8 или красная КМ на эталоне; иначе жёлтый. "
        "Серый: средняя близость < 0,7, доля непокрытых > 30 %, недостаточно данных "
        "или неизвестна шкала метрики. Пороги можно скорректировать в настройках.",
        reason=pre.get("reason") or ("Оценка недоступна или недостаточно надёжна; возможен информационный режим."
                                    if res["report"]["semaphore"] in ("gray", "grey") else ""),
    )


def report_valtest_local_drift(res, semaphore_title):
    semaphore_color = {"yellow": "amber", "grey": "gray"}.get(
        res["report"]["semaphore"], res["report"]["semaphore"])
    html_report = html_report_valtest_local_drift(res, semaphore_title)
    pre = res.get("precomputed", {})
    def number(value):
        return float(value) if value is not None and math.isfinite(float(value)) else None

    metric_value = number(pre.get("metric_value"))
    metric_estimate = number(pre.get("metric_value_estimate"))
    reliability = {key: number(value) for key, value in pre.get("reliability", {}).items()}
    return {
        "all_results": {
            "calculated_traffic_lights": {
                "test_light": semaphore_color,
                "semaphore_title": semaphore_title,
            },
            "color": semaphore_color,
            "status": "not_computable" if semaphore_color == "gray" else "computed",
            "reason": pre.get("reason"),
            "metric_value": metric_value,
            "metric_value_estimate": metric_estimate,
            "drop_estimate": (
                metric_value - metric_estimate
                if metric_value is not None
                and metric_estimate is not None
                and not pd.isna(metric_estimate)
                else None
            ),
            "reliability_mean": reliability.get("mean"),
            "share_uncovered": reliability.get("share_below_threshold"),
        },
        "hidden_port": html_report,
    }


# P0-3: ключи унифицированы на "gray"
_SEMAPHORE_TITLE = {
    "red": "Результат теста локального дрифта соответствует красному светофору",
    "green": "Результат теста локального дрифта соответствует зелёному светофору",
    "yellow": "Результат теста локального дрифта соответствует жёлтому светофору",
    "gray": "Результат теста локального дрифта не может быть оценён",
}


# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# =============================================================================

def main(
    reference_umr: pd.DataFrame,
    monitoring_umr: pd.DataFrame,
    monitoring_metric: dict,
    ann_config: Any = None,
    n_closest: int = 5,
    metric_agg: str = "single_mean",
    data_types: tuple = ("train", "test"),
    green_threshold: float = 0.5,
    red_threshold: float = 0.8,
    reliability_threshold: float = 0.7,
    greater_is_better: bool = True,
    is_info: bool = False,
):
    """
    Запуск теста на локальный дрифт.

    Изменения относительно baseline:
    - P0-3: ключи словарей унифицированы на "gray"
    - P0-6: main_metric корректно перезаписывается после rename
    - P1-6: literal_eval защищён от уже-готового dict/tuple
    - P1-7: dropna с subset
    """
    # Защитный literal_eval
    if isinstance(data_types, str):
        data_types = literal_eval(data_types)

    if ann_config is None:
        ann_config = {"create_index": {"exact": True}, "search_query": {}}
    elif isinstance(ann_config, str):
        ann_config = literal_eval(ann_config)

    # P0-5: пороги в правильном порядке
    semaphore_threshold = (
        min(red_threshold, green_threshold),
        max(red_threshold, green_threshold),
    )

    reference_frame, monitoring_frame = prepare_drift_frames(
        reference_umr, monitoring_umr, monitoring_metric
    )
    main_metric = "target"

    sampler = AutoAsessorSampler(agent_df=monitoring_frame, real_df=reference_frame)
    scorer = AutoAsessorScorer(metrics=METRICS)

    config = Config()
    embedding_model = GigaEmbed(**config.contour_configs)

    ann = ANN()

    logging.info("Тест на локальный дрифт запущен")
    res = valtest_local_drift_stability(
        sampler=sampler,
        scorer=scorer,
        main_metric=main_metric,
        model=embedding_model,
        ann=ann,
        ann_config=ann_config,
        n_closest=n_closest,
        metric_binarizer=None,
        metric_agg=metric_agg,
        data_types=data_types,
        semaphore_threshold=semaphore_threshold,
        reliability_threshold=reliability_threshold,
        greater_is_better=greater_is_better,
        is_info=is_info,
        test_color=None,
        metric_value_estimate=None,
        reliability_stats=None,
        metric_scale=monitoring_metric.get("baseline", {}).get("scale"),
    )
    logging.info(res)

    semaphore_color = res["report"]["semaphore"]
    semaphore_title = _SEMAPHORE_TITLE[semaphore_color]

    report_result = report_valtest_local_drift(res, semaphore_title)
    report_result["all_results"]["test_name"] = "local_drift"

    return {
        "all_results": report_result["all_results"],
        "test_description": report_result["hidden_port"],
    }
