"""
Скореры для подсчета метрик качества.

Базовый класс Scorer определяет интерфейс для подсчета метрик.
AutoAsessorScorer вычисляет среднее значение целевой переменной (target).
"""

import typing as tp
from abc import ABC, abstractmethod

from llm_val.sampler import Sampler


# =============================================================================
# БАЗОВЫЙ КЛАСС СКОРЕРА
# =============================================================================

class Scorer(ABC):
    """
    Абстрактный класс скорера.
    Все скореры должны наследоваться от данного класса.
    """

    # -------------------------------------------------------------------------
    # Инициализация
    # -------------------------------------------------------------------------

    def __init__(self, metrics: tp.Dict[str, tp.Any]):
        self.metrics = metrics

    # -------------------------------------------------------------------------
    # Абстрактный метод
    # -------------------------------------------------------------------------

    @abstractmethod
    def calc(
        self, sampler: Sampler, data_type: str, metric_name: str
    ) -> tp.Dict[str, float]:
        """Метод подсчета метрик"""
        raise NotImplementedError(f"Определите calc в {self.__class__.__name__}")


# =============================================================================
# СКОРЕР ДЛЯ АВТОАССЕССОРА
# =============================================================================

class AutoAsessorScorer(Scorer):
    """
    Скорер, который считает метрики по столбцу 'agent_target'.
    Ключевая метрика — среднее значение agent_target.
    """

    # -------------------------------------------------------------------------
    # Подсчет метрик
    # -------------------------------------------------------------------------

    def calc(
        self, sampler: Sampler, data_type: str, metric_name: str | None = None
    ) -> tp.Dict[str, float]:
        """
        Подсчёт метрик для указанного типа данных.
        Поддерживается только одна метрика: 'target' — среднее по target.
        """
        metrics = self.metrics
        data = getattr(sampler, data_type)["y"]
        if metric_name is not None:
            metrics = {metric_name: self.metrics[metric_name]}
        metric_res = {}

        for func_name, func_dict in metrics.items():
            if func_dict["is_singlecol"]:
                for name_col in data:
                    name = f"{name_col}"
                    metric_res[name] = func_dict["call"](data[name_col])
            else:
                metric_res[func_name] = func_dict["call"](data)

        return metric_res
