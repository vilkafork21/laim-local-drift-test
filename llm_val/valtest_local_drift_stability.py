"""
Тест на локальный дрифт запросов.

Для каждого OOT-запроса находятся top-N семантически ближайших OOS-запросов
(через ANN). Ожидаемое качество ответа считается как взвешенное по similarity
среднее меток качества соседей. Параллельно считается reliability — мера
надёжности оценки. Если reliability низкая — тест неинформативен (gray).
"""

import logging
import typing as tp
from copy import deepcopy
from decimal import Decimal

import numpy as np
import pandas as pd
from llm_val.report_helper import semaphore_by_threshold, worst_semaphore
from llm_val.sampler import AutoAsessorSampler as Sampler
from llm_val.scorer import AutoAsessorScorer as Scorer
from llm_val.valtest_metric import valtest_metric


# Минимальный размер OOS для информативного теста
MIN_OOS_SAMPLES = 100

# Адаптивное правило для n_closest по размеру OOS (P2-1)
def _adaptive_n_closest(n_oos: int, user_n_closest: int) -> int:
    upper = max(3, min(10, n_oos // 20))
    return min(user_n_closest, upper) if user_n_closest > 0 else upper


# =============================================================================
# ФОРМИРОВАНИЕ ОТЧЕТА
# =============================================================================

def report_valtest_local_drift_stability(
    metric_value: tp.Dict[str, float],
    metric_value_estimate: float,
    reliability_stats: tp.Dict[str, float],
    main_metric: str,
    test_color: str,
    data_types: tp.Tuple[str, str] = ("train", "test"),
    semaphore_threshold: tp.Tuple[float, float] = (0.5, 0.8),
    reliability_threshold: float = 0.7,
    greater_is_better: bool = True,
    is_info: bool = False,
) -> tp.Dict[str, tp.Any]:
    """
    Создание отчёта по результатам теста.

    Reliability теперь содержит словарь со статистиками распределения (P1-9, P2-6).
    """
    logging.info("Начало формирования отчёта")
    metric_value_scalar = metric_value[main_metric]
    reliability_mean = reliability_stats["mean"]

    # Условия серого светофора: явный is_info ИЛИ низкая средняя надёжность
    # ИЛИ слишком много «непокрытых» запросов (P1-9)
    is_gray = (
        is_info
        or reliability_mean < reliability_threshold
        or reliability_stats.get("share_below_threshold", 0.0) > 0.3
        or not np.isfinite(metric_value_estimate)
        or not np.isfinite(reliability_mean)
    )

    abs_diff = float(Decimal(str(metric_value_estimate)) - Decimal(str(metric_value_scalar)))
    if greater_is_better:
        abs_diff = -abs_diff

    if is_gray:
        result_color = "gray"
    else:
        color_by_metric = semaphore_by_threshold(
            abs_diff, semaphore_threshold, greater_is_better=False
        )
        result_color = worst_semaphore([test_color, color_by_metric])

    df = pd.DataFrame(
        {
            f"Значение метрики на {data_types[0]}": [round(float(metric_value_scalar), 4)],
            f"Оценённое значение метрики на {data_types[1]}": [
                round(float(metric_value_estimate), 4) if not np.isnan(metric_value_estimate) else None
            ],
            "Абсолютная разница": [round(float(metric_value_estimate - metric_value_scalar), 4)
                                   if not np.isnan(metric_value_estimate) else None],
            "Надёжность (среднее)": [round(float(reliability_stats["mean"]), 4)],
            "Надёжность (медиана)": [round(float(reliability_stats["median"]), 4)],
            "Надёжность (q05)": [round(float(reliability_stats["q05"]), 4)],
            "Доля непокрытых запросов": [round(float(reliability_stats["share_below_threshold"]), 4)],
            "Результат теста": [result_color],
        }
    )

    return {
        "semaphore": result_color,
        "result_plots": [],
        "result_dataframes": [df],
    }


# =============================================================================
# ОСНОВНОЙ ТЕСТ
# =============================================================================

def valtest_local_drift_stability(
    sampler: Sampler,
    scorer: Scorer,
    main_metric: str,
    model: tp.Any,
    ann: tp.Any,
    ann_config: tp.Dict[str, tp.Any],
    n_closest: int = 5,
    metric_binarizer: tp.Optional[tp.Callable] = None,
    metric_agg: str = "single_mean",
    data_types: tp.Tuple[str, str] = ("train", "test"),
    semaphore_threshold: tp.Tuple[float, float] = (0.5, 0.8),
    reliability_threshold: float = 0.7,
    greater_is_better: bool = True,
    is_info: bool = False,
    metric_value: tp.Optional[tp.Dict[str, float]] = None,
    test_color: tp.Optional[str] = None,
    metric_value_estimate: tp.Optional[float] = None,
    reliability_stats: tp.Optional[tp.Dict[str, float]] = None,
    metric_scale: str = "ratio",
    **kwargs,
) -> tp.Dict[str, tp.Any]:
    """
    Тест на анализ качества ответа модели в зависимости от локального дрифта запросов.

    Изменения относительно baseline:
    - P0-1: setattr использует oot["X"], а не oos["X"]
    - P0-2: мутация y_oos[y_oos==0]=-1 переведена в копию и вынесена ИЗ цикла
    - P0-6: main_metric после rename корректно работает (см. main.py)
    - P1-2: финальный score клиппуется в [0, 1]
    - P1-3: y_oos переведён в numpy для скоростного доступа
    - P1-9: reliability — словарь {mean, median, q05, share_below_threshold}
    - P2-1: адаптивный n_closest по размеру OOS
    - P2-2: поддержка не-бинарных меток через явное центрирование
    """
    oos = getattr(sampler, data_types[0])
    oot = getattr(sampler, data_types[1])
    sampler_copy = deepcopy(sampler)

    X_oos, y_oos = oos["X"], oos["y"].copy()  # copy, чтобы не мутировать sampler (P0-2)
    X_oot, y_oot = oot["X"], oot["y"].copy()

    if metric_binarizer is not None:
        y_oos = pd.DataFrame({"metric_value": metric_binarizer(y_oos)})
        y_oot_binarized = pd.DataFrame({"metric_value": metric_binarizer(y_oot)})
        setattr(sampler_copy, data_types[0],
                {"X": oos["X"], "y": y_oos})
        # P0-1: для OOT используем oot["X"], а не oos["X"]
        setattr(sampler_copy, data_types[1],
                {"X": oot["X"], "y": y_oot_binarized})

    n_oos = len(X_oos)
    labels = np.asarray(y_oos.values, dtype=float).ravel()
    reason = None
    if n_oos < MIN_OOS_SAMPLES:
        reason = f"Недостаточно объектов OOS: {n_oos} < {MIN_OOS_SAMPLES}"
    elif len(X_oot) == 0:
        reason = "Нет запросов мониторинга"
    elif not np.isfinite(labels).all() or np.any((labels < 0) | (labels > 1)):
        reason = "Метки OOS должны быть конечными числами в известной шкале [0; 1]"
    elif metric_scale != "ratio" and not np.isin(labels, [0, 1]).all():
        reason = "Для непрерывной метрики raw не задан известный диапазон; прогноз не вычисляется"
    def unavailable(message: str) -> dict:
        reference_value = metric_value or {main_metric: float(labels.mean()) if labels.size and np.isfinite(labels).all() else np.nan}
        logging.warning(message)
        empty_stats = {"mean": np.nan, "median": np.nan, "q05": np.nan, "share_below_threshold": 1.0}
        report = report_valtest_local_drift_stability(
            reference_value,
            np.nan, empty_stats, main_metric,
            test_color or "gray", data_types, semaphore_threshold,
            reliability_threshold, greater_is_better, is_info=True,
        )
        return {
            "report": report,
            "precomputed": {
                "metric_value": (reference_value)[main_metric],
                "metric_value_estimate": np.nan,
                "reliability": empty_stats,
                "reason": message,
            },
        }

    if reason:
        return unavailable(reason)

    # P2-1: адаптируем n_closest под размер OOS
    n_closest = _adaptive_n_closest(n_oos, n_closest)
    logging.info(f"Используется n_closest={n_closest} для OOS размера {n_oos}")

    if metric_value is None or test_color is None:
        logging.info("Расчёт основной метрики на OOS")
        metric_result = valtest_metric(
            sampler=sampler_copy,
            scorer=scorer,
            main_metric=main_metric,
            data_type=data_types[0],
            metric_agg=metric_agg,
            greater_is_better=greater_is_better,
        )
        test_color = metric_result["report"]["semaphore"]
        metric_value = metric_result["precomputed"]["metric_value"]

    if metric_value_estimate is None or reliability_stats is None:
        logging.info("Подготовка ANN и эмбеддингов")

        # P2-4: фильтруем пустые question
        oos_q = X_oos["question"].astype(str).str.slice(0, 1000).tolist()
        oot_q = X_oot["question"].astype(str).str.slice(0, 1000).tolist()
        if any(not q.strip() for q in oos_q):
            logging.warning("Некоторые OOS-вопросы пусты; заменяются заглушкой")
            oos_q = [q if q.strip() else "<empty>" for q in oos_q]
        if any(not q.strip() for q in oot_q):
            logging.warning("Некоторые OOT-вопросы пусты; заменяются заглушкой")
            oot_q = [q if q.strip() else "<empty>" for q in oot_q]

        oos_embeddings = np.asarray(model.get_embedding(oos_q), dtype=np.float32)
        if not np.isfinite(oos_embeddings).all():
            return unavailable("Эмбеддинги OOS содержат невалидные значения")

        # ANN с автоматическим fallback на exact при малой выборке (P1-4 внутри ann.py)
        ann.create_index(oos_embeddings, **ann_config.get("create_index", {}))

        oot_embeddings = np.asarray(model.get_embedding(oot_q), dtype=np.float32)
        if not np.isfinite(oot_embeddings).all():
            return unavailable("Эмбеддинги OOT содержат невалидные значения")

        # Диапазон [0; 1] задан контрактом, а не минимумом/максимумом выборки.
        y_oos_signed = 2.0 * labels - 1.0

        test_scores: tp.List[float] = []
        reliability_values: tp.List[float] = []
        search_kwargs = ann_config.get("search_query", {})

        for query_emb in oot_embeddings:
            query_res = ann.search_query(query_emb, n_closest, **search_kwargs)
            sims = np.asarray([item["similarity"] for item in query_res], dtype=float)
            ids = np.asarray([item["id"] for item in query_res], dtype=int)
            # Защита от пустого ответа ANN (теоретически невозможно, но хедж)
            if len(sims) == 0:
                reliability_values.append(0.0)
                test_scores.append(0.5)
                continue

            reliability_values.append(float(np.mean(sims)))
            # P1-2: клипуем итоговый score в [0, 1]
            raw = 0.5 + float(np.sum(sims * y_oos_signed[ids])) / (2.0 * len(sims))
            test_scores.append(float(np.clip(raw, 0.0, 1.0)))

        test_scores_arr = np.asarray(test_scores, dtype=float)
        reliability_arr = np.asarray(reliability_values, dtype=float)

        metric_value_estimate = float(np.mean(test_scores_arr))
        # P1-9: статистики распределения reliability
        reliability_stats = {
            "mean": float(np.mean(reliability_arr)),
            "median": float(np.median(reliability_arr)),
            "q05": float(np.quantile(reliability_arr, 0.05)),
            "share_below_threshold": float(np.mean(reliability_arr < reliability_threshold)),
        }

    precomputed = {
        "metric_value": metric_value[main_metric],
        "metric_value_estimate": metric_value_estimate,
        "reliability": reliability_stats,
        "n_closest": n_closest,
        "reliability_threshold": reliability_threshold,
        "semaphore_threshold": semaphore_threshold,
    }
    report = report_valtest_local_drift_stability(
        metric_value=metric_value,
        metric_value_estimate=metric_value_estimate,
        reliability_stats=reliability_stats,
        main_metric=main_metric,
        test_color=test_color,
        data_types=data_types,
        semaphore_threshold=semaphore_threshold,
        reliability_threshold=reliability_threshold,
        greater_is_better=greater_is_better,
        is_info=is_info,
    )
    if report["semaphore"] == "gray":
        precomputed["reason"] = (
            "Информационный режим" if is_info else
            "Недостаточная надёжность прогноза: средняя близость ниже порога, "
            "доля непокрытых запросов выше 30 % или оценка невалидна"
        )
        logging.warning(precomputed["reason"])
    return {"report": report, "precomputed": precomputed}
