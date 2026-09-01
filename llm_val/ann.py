"""
ANN (Approximate Nearest Neighbor) поиск с использованием FAISS.

Изменения относительно baseline:
- Авто-fallback на exact-индекс при малой выборке (P1-4)
- `search_query` принимает **kwargs (P1-8)
- `n_partitions` / `nprobe` параметризованы через ann_config (P2-3)
"""

import logging
from typing import Any, List, Dict

import faiss
import numpy as np


# Минимальное число точек на партицию для адекватного обучения IVF
MIN_POINTS_PER_PARTITION = 40


class ANN:
    """Класс для приближённого поиска ближайших соседей через FAISS."""

    def __init__(self, n_partitions: int = 10, nprobe: int = 2):
        self.n_partitions = n_partitions  # Число кластеров в IVF индексации
        self.nprobe = nprobe
        self.index = None
        self.exact = False

    def create_index(
        self,
        embedding: Any,
        exact: bool = False,
        n_partitions: int = None,
        nprobe: int = None,
        **kwargs,
    ):
        """
        Создаёт индекс для поиска.

        Если `exact=True` или выборка слишком мала для IVF (n < MIN_POINTS_PER_PARTITION * n_partitions),
        автоматически переключается на точный поиск IndexFlatIP (P1-4).
        """
        if n_partitions is not None:
            self.n_partitions = int(n_partitions)
        if nprobe is not None:
            self.nprobe = int(nprobe)

        db_vectors = np.array(embedding, dtype=np.float32)
        if db_vectors.ndim != 2:
            raise ValueError(f"embeddings должны быть 2D, получено {db_vectors.shape}")
        n, dimension = db_vectors.shape
        faiss.normalize_L2(db_vectors)

        needs_exact = exact or n < MIN_POINTS_PER_PARTITION * self.n_partitions
        if needs_exact and not exact:
            logging.info(
                f"ANN: выборка {n} слишком мала для IVF с {self.n_partitions} партициями; "
                "fallback на exact (IndexFlatIP)"
            )
        self.exact = needs_exact

        if self.exact:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        else:
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, dimension, self.n_partitions, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(db_vectors)
            self.index.nprobe = self.nprobe
        self.index.add_with_ids(db_vectors, np.arange(len(db_vectors)))

    def search_query(self, query_vector: Any, k: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Поиск top-k ближайших соседей.
        P1-8: принимает **kwargs (игнорирует), чтобы не падать при расширении ann_config.
        """
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        similarities, ids = self.index.search(query_vector, k)
        return [
            {"id": int(ids[0][i]), "similarity": float(similarities[0][i])}
            for i in range(len(ids[0]))
            if ids[0][i] >= 0  # FAISS может вернуть -1 для незаполненных слотов
        ]
