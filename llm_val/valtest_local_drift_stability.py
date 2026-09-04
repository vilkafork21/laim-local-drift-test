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

import numpy as np
import pandas as pd
from llm_val.sampler import AutoAsessorSampler as Sampler
from llm_val.scorer import AutoAsessorScorer as Scorer
from llm_val.valtest_metric import valtest_metric

logger = logging.getLogger(__name__)


# Минимальный размер OOS для информативного теста
MIN_OOS_SAMPLES = 30

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
    data_types: tp.Tuple[str, str] = ("train", "test"),
    reliability_threshold: float = 0.2,
    uncovered_amber: float = 0.3,
    uncovered_red: float = 0.5,
    is_info: bool = False,
) -> tp.Dict[str, tp.Any]:
    """Отчёт теста покрытия потока эталоном (карточка 6.3.7).

    Цвет отражает тяжесть непокрытой доли потока; оценка метрики по соседям
    публикуется как информация и на цвет не влияет."""
    logger.info("Начало формирования отчёта")
    metric_value_scalar = metric_value[main_metric]
    reliability_mean = reliability_stats["mean"]
    share_uncovered = reliability_stats.get("share_below_threshold", 0.0)

    if is_info or np.isnan(metric_value_estimate) or np.isnan(reliability_mean):
        result_color = "gray"
    elif share_uncovered > uncovered_red:
        result_color = "red"
    elif share_uncovered > uncovered_amber or reliability_mean < reliability_threshold:
        result_color = "yellow"
    else:
        result_color = "green"

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
    reliability_threshold: float = 0.2,
    uncovered_amber: float = 0.3,
    uncovered_red: float = 0.5,
    min_oos_samples: int = MIN_OOS_SAMPLES,
    min_oot_samples: int = 30,
    greater_is_better: bool = True,
    is_info: bool = False,
    metric_value: tp.Optional[tp.Dict[str, float]] = None,
    metric_value_estimate: tp.Optional[float] = None,
    reliability_stats: tp.Optional[tp.Dict[str, float]] = None,
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

    # Минимумы объёма с обеих сторон: без них цвет выставлялся по одной единице.
    n_oos = len(X_oos)
    n_oot = len(X_oot)
    if n_oos < min_oos_samples or n_oot < min_oot_samples:
        if n_oos < min_oos_samples:
            reason_code = "insufficient_reference_units"
            reason = f"OOS {n_oos} единиц меньше минимума {min_oos_samples}"
        else:
            reason_code = "insufficient_monitoring_units"
            reason = f"OOT {n_oot} единиц меньше минимума {min_oot_samples}"
        logger.warning("Тест покрытия не оценивается: %s", reason)
        empty_stats = {"mean": np.nan, "median": np.nan, "q05": np.nan, "share_below_threshold": 1.0}
        report = report_valtest_local_drift_stability(
            metric_value or {main_metric: np.nan},
            np.nan, empty_stats, main_metric, data_types,
            reliability_threshold, uncovered_amber, uncovered_red, is_info=True,
        )
        return {
            "report": report,
            "precomputed": {
                "metric_value": (metric_value or {main_metric: np.nan})[main_metric],
                "metric_value_estimate": np.nan,
                "reliability": empty_stats,
                "reason_code": reason_code,
                "reason": reason,
                "n_oos": int(n_oos),
                "n_oot": int(n_oot),
                "n_closest": None,
            },
        }

    # P2-1: адаптируем n_closest под размер OOS
    n_closest = _adaptive_n_closest(n_oos, n_closest)
    logger.info(f"Используется n_closest={n_closest} для OOS размера {n_oos}")

    if metric_value is None:
        logger.info("Расчёт основной метрики на OOS")
        metric_result = valtest_metric(
            sampler=sampler_copy,
            scorer=scorer,
            main_metric=main_metric,
            data_type=data_types[0],
            metric_agg=metric_agg,
            greater_is_better=greater_is_better,
        )
        # Светофор уровня метрики из valtest_metric не используется: уровень
        # качества корзины — предмет валидации, а не теста покрытия.
        metric_value = metric_result["precomputed"]["metric_value"]

    if metric_value_estimate is None or reliability_stats is None:
        logger.info("Подготовка ANN и эмбеддингов")

        # P2-4: фильтруем пустые question
        oos_q = X_oos["question"].astype(str).tolist()
        oot_q = X_oot["question"].astype(str).tolist()
        if any(not q.strip() for q in oos_q):
            logger.warning("Некоторые OOS-вопросы пусты; заменяются заглушкой")
            oos_q = [q if q.strip() else "<empty>" for q in oos_q]
        if any(not q.strip() for q in oot_q):
            logger.warning("Некоторые OOT-вопросы пусты; заменяются заглушкой")
            oot_q = [q if q.strip() else "<empty>" for q in oot_q]

        oos_embeddings = np.asarray(model.get_embedding(oos_q), dtype=np.float32)
        if np.isnan(oos_embeddings).any():
            oos_embeddings = np.nan_to_num(oos_embeddings, nan=0.0)

        # ANN с автоматическим fallback на exact при малой выборке (P1-4 внутри ann.py)
        ann.create_index(oos_embeddings, **ann_config.get("create_index", {}))

        oot_embeddings = np.asarray(model.get_embedding(oot_q), dtype=np.float32)
        if np.isnan(oot_embeddings).any():
            oot_embeddings = np.nan_to_num(oot_embeddings, nan=0.0)

        # P0-2: трансформация меток ВНЕ цикла; работаем с numpy (P1-3)
        # P2-2: для бинарных {0,1} меток → {-1,+1}; для других → центрируем
        y_oos_arr = np.asarray(y_oos.values, dtype=float).ravel()
        unique_vals = np.unique(y_oos_arr[~np.isnan(y_oos_arr)])
        label_span = None
        if set(unique_vals.tolist()).issubset({0.0, 1.0}):
            y_oos_signed = np.where(y_oos_arr == 0, -1.0, 1.0)
        else:
            # Центрирование: y' = 2*(y - min)/(max - min) - 1
            y_min, y_max = float(np.nanmin(y_oos_arr)), float(np.nanmax(y_oos_arr))
            label_span = (y_min, y_max)
            if y_max > y_min:
                y_oos_signed = 2.0 * (y_oos_arr - y_min) / (y_max - y_min) - 1.0
            else:
                y_oos_signed = np.zeros_like(y_oos_arr)
            logger.info(f"Не-бинарные метки в диапазоне [{y_min}, {y_max}] — центрирование")

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
        if label_span is not None:
            # score лежит в [0, 1] относительно центрированных меток; metric_value
            # считается в шкале меток корзины — возвращаем оценку в ту же шкалу,
            # иначе снижение сравнивает разные шкалы.
            y_min, y_max = label_span
            metric_value_estimate = y_min + metric_value_estimate * (y_max - y_min)
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
        "n_oos": int(n_oos),
        "n_oot": int(n_oot),
        "n_closest": int(n_closest),
    }
    report = report_valtest_local_drift_stability(
        metric_value=metric_value,
        metric_value_estimate=metric_value_estimate,
        reliability_stats=reliability_stats,
        main_metric=main_metric,
        data_types=data_types,
        reliability_threshold=reliability_threshold,
        uncovered_amber=uncovered_amber,
        uncovered_red=uncovered_red,
        is_info=is_info,
    )
    return {"report": report, "precomputed": precomputed}
