"""
Utility functions.
"""

import numpy as np


METRICS = {
    "single_mean": {"call": lambda x: float(np.mean(x.values)), "is_singlecol": True},
    "multicol_mean": {"call": lambda x: float(np.mean(x.values)), "is_singlecol": False},
}


def string_to_float(value):
    """
    P3-3: специфичные исключения вместо bare except.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
