"""
Тест на локальный дрифт запросов — точка входа.
"""

import logging
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

from html_report_helper import display_semaphore, show_criteria_semaphore


# =============================================================================
# ФУНКЦИИ ФОРМИРОВАНИЯ ОТЧЕТОВ
# =============================================================================

def _table_styles():
    return [
        {"selector": "th", "props": [
            ("background-color", "#f5f5f5"),
            ("text-align", "center"),
            ("border", "1px solid #ddd"),
            ("padding", "5px"),
        ]},
        {"selector": "td", "props": [
            ("text-align", "left"),
            ("border", "1px solid #ddd"),
            ("padding", "5px"),
        ]},
        {"selector": "", "props": [
            ("border-collapse", "collapse"),
            ("border", "1px solid black"),
        ]},
    ]


def html_report_valtest_local_drift(res, semaphore_title):
    table_styles = _table_styles()

    green_criterion = (
        "Абсолютное снижение ключевой метрики менее 15 п.п. "
        "И светофор OOT — «Зелёный»"
    )
    yellow_criterion = "Иначе"
    red_criterion = (
        "Абсолютное снижение ключевой метрики качества более 25 п.п. "
        "ИЛИ светофор OOT — «Красный»"
    )
    grey_criterion = (
        "Средний scoreANNi &lt; 0.2 ИЛИ доля непокрытых OOT-запросов &gt; 30%"
    )

    criterion_df = show_criteria_semaphore(
        green_criterion, yellow_criterion, red_criterion, grey_criterion, table_styles
    )
    criterion_df_html = criterion_df.to_html(border=0, classes="table")

    semaphore_color = res["report"]["semaphore"]
    semaphore_html = display_semaphore(semaphore_color, return_html=True)

    pre = res["precomputed"]
    reliability_stats = pre.get("reliability", {})
    mv = pre.get("metric_value")
    mve = pre.get("metric_value_estimate")

    def _fmt(v):
        if v is None:
            return "n/a"
        try:
            if pd.isna(v):
                return "n/a"
        except (TypeError, ValueError):
            pass
        return f"{float(v):.3f}"

    res_df = pd.DataFrame(
        {
            "Показатель": [
                "Значение метрики на валидации",
                "Оценка метрики на мониторинге",
                "Абсолютное снижение",
                "Надёжность (среднее)",
                "Надёжность (медиана)",
                "Надёжность (q05)",
                "Доля непокрытых запросов",
                "Результат теста",
            ],
            "Значение": [
                _fmt(mv),
                _fmt(mve),
                _fmt(abs(mv - mve)) if (mv is not None and mve is not None and not pd.isna(mve)) else "n/a",
                _fmt(reliability_stats.get("mean")),
                _fmt(reliability_stats.get("median")),
                _fmt(reliability_stats.get("q05")),
                _fmt(reliability_stats.get("share_below_threshold")),
                semaphore_html,
            ],
        }
    )

    try:
        res_df_to_html = res_df.style.hide().set_table_styles(table_styles)
    except AttributeError:
        res_df_to_html = res_df.style.hide_index().set_table_styles(table_styles)
    res_df_html = res_df_to_html.to_html(border=0, classes="table")

    html_report = f"""
<h2 style="text-align: center;">Тест на локальный drift запросов</h2>
<p style="text-align: left;"><b>Цель теста</b></p>
<p style="text-align: left;">Оценить ключевую метрику качества агента исходя из степени изменения запросов к ней.</p>
<p style="text-align: left;">При существовании локального дрифта запросов агент должен показывать уровень качества, соизмеримый с качеством на первичной валидации.</p>
<p style="text-align: left;"><b>Условия проведения</b></p>
<ul style="text-align: left; margin-left: 20px; padding-left: 20px;">
    <li>Для СЗ &gt; E</li>
    <li>Минимум {MIN_OOS_SAMPLES} наблюдений в OOS</li>
    <li>Метки качества бинарные {{0, 1}} или непрерывные в известном диапазоне</li>
</ul>
<p style="text-align: left;"><b>Алгоритм расчёта</b></p>
<ol style="text-align: left; margin-left: 20px; padding-left: 20px;">
    <li>Для каждого экземпляра запроса из OOT подбирается N наиболее близких запросов из OOS через ANN на эмбеддингах GigaChat.</li>
    <li>Ожидаемое качество ответа для OOT-запроса считается по формуле:</li>
</ol>
<p style="text-align: left; margin-left: 20px; padding-left: 20px;">
    score<sub>j</sub> = 0.5 + (1 / (2·N)) · &sum;<sup>N</sup><sub>i=1</sub> scoreANN<sub>i</sub> · target<sub>i</sub><br>
    где target<sub>i</sub> = {{-1, +1}} для бинарных меток или 2·(y − y<sub>min</sub>)/(y<sub>max</sub> − y<sub>min</sub>) − 1 для непрерывных.
</p>
<ol style="text-align: left; margin-left: 20px; padding-left: 20px; counter-reset: list 2;">
    <li>Финальное качество — среднее score<sub>j</sub> по OOT, ограниченное [0, 1].</li>
    <li>Надёжность: среднее, медиана, 5-й перцентиль similarity top-N и доля «непокрытых» запросов.</li>
</ol>
<p style="text-align: left;"><b>Критерии выставления светофора</b></p>
<div style="text-align: left; width: 100%;">{criterion_df_html}</div><br>
<p style="text-align: left;"><b>Результаты теста</b></p>
<div style="text-align: left; width: 100%;">{res_df_html}</div><br>
"""
    return html_report


# Внутренний словарь теста — red/yellow/green/gray; платформа и laim-agg
# ждут amber: нормализуем на границе вывода, как соседние тесты (LAIM-0004).
_PLATFORM_COLOR = {"yellow": "amber", "grey": "gray"}


def _gray_reason(pre: dict, reliability_threshold: float, is_info: bool) -> str | None:
    """Причина серого светофора словами — ту же логику применяет отчёт."""
    reliability = pre.get("reliability", {})
    estimate = pre.get("metric_value_estimate")
    if is_info:
        return "информационный режим (is_info)"
    if estimate is None or pd.isna(estimate):
        return f"оценка недоступна: OOS меньше {MIN_OOS_SAMPLES} единиц или нет соседей"
    if reliability.get("share_below_threshold", 0.0) > 0.3:
        return "доля запросов без надёжных соседей выше 0.3"
    if (reliability.get("mean") or 0.0) < reliability_threshold:
        return f"средняя надёжность оценки ниже порога {reliability_threshold}"
    return None


def report_valtest_local_drift(res, semaphore_title, reliability_threshold=0.2, is_info=False):
    semaphore_color = _PLATFORM_COLOR.get(res["report"]["semaphore"], res["report"]["semaphore"])
    html_report = html_report_valtest_local_drift(res, semaphore_title)
    pre = dict(res.get("precomputed", {}))
    if semaphore_color == "gray" and not pre.get("reason"):
        pre["reason"] = _gray_reason(pre, reliability_threshold, is_info)
    metric_value = pre.get("metric_value")
    metric_estimate = pre.get("metric_value_estimate")
    reliability = pre.get("reliability", {})
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
                abs(metric_value - metric_estimate)
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
    "red": "Локальный дрифт запросов: ожидаемое качество на новых запросах "
           "существенно ниже валидационного — красный светофор",
    "green": "Локальный дрифт запросов: ожидаемое качество на новых запросах "
             "соответствует валидационному — зелёный светофор",
    "yellow": "Локальный дрифт запросов: ожидаемое качество на новых запросах "
              "заметно ниже валидационного — жёлтый светофор",
    "gray": "Локальный дрифт запросов не может быть оценён",
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
    green_threshold: float = 0.15,
    red_threshold: float = 0.25,
    reliability_threshold: float = 0.2,
    greater_is_better: bool = True,
    is_info: bool = False,
):
    """
    Запуск теста на локальный дрифт.

    Изменения относительно baseline:
    - P0-3: ключи словарей унифицированы на "gray"
    - P0-4: reliability_threshold default = 0.2 (как в HTML)
    - P0-5: red/green defaults = 0.25/0.15 (как в HTML)
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
    )
    logging.info(res)

    semaphore_color = res["report"]["semaphore"]
    semaphore_title = _SEMAPHORE_TITLE[semaphore_color]

    report_result = report_valtest_local_drift(
        res, semaphore_title, reliability_threshold=reliability_threshold, is_info=is_info)
    report_result["all_results"]["test_name"] = "local_drift"

    return {
        "all_results": report_result["all_results"],
        "test_description": report_result["hidden_port"],
    }
