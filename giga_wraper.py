"""
Обёртка над GigaChat для работы с эмбеддингами.

Батчинг и retry — копия из global drift (P1-5).

Защита от лимита эмбеддера GigaChat (514 токенов на один вход):
  - длинный вход делится по max_chars, embeddings частей усредняются;
  - при 413 части делятся повторно без потери хвоста.

Сетевые/транзиентные ошибки по-прежнему повторяются с backoff (max_chars их не трогает).
"""

import logging
import time
from typing import Iterable, List, Optional

import numpy as np
from gigachat import GigaChat

logger = logging.getLogger(__name__)


DEFAULT_BATCH_SIZE = 100
DEFAULT_RETRIES = 3
DEFAULT_BASE_BACKOFF = 1.0
# Лимит эмбеддера GigaChat — 514 токенов на вход. Для русского ~2.5-3 симв./токен,
# поэтому 1000 символов ≈ 350-450 токенов — с запасом ниже лимита.
DEFAULT_MAX_CHARS = 1000
# Нижняя граница реактивного ужатия (200 симв. — заведомо безопасно по токенам).
DEFAULT_MIN_CHARS = 200


class GigaEmbed(GigaChat):
    def __init__(self, *args, batch_size: int = DEFAULT_BATCH_SIZE,
                 retries: int = DEFAULT_RETRIES, base_backoff: float = DEFAULT_BASE_BACKOFF,
                 max_chars: int = DEFAULT_MAX_CHARS, min_chars: int = DEFAULT_MIN_CHARS,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_size = batch_size
        self._retries = retries
        self._base_backoff = base_backoff
        self._max_chars = int(max_chars)
        self._min_chars = int(min_chars)

    @staticmethod
    def _is_token_limit_error(err: Exception) -> bool:
        """Признак ошибки лимита токенов эмбеддера (а не сетевой)."""
        s = str(err).lower()
        return ("413" in s) or ("tokens limit exceeded" in s) or ("payload too large" in s)

    def get_embedding(self, list_of_texts: Iterable[str]) -> np.ndarray:
        texts = ["" if t is None else str(t) for t in list_of_texts]
        if not texts:
            return np.zeros((0, 0), dtype=float)

        chunked = sum(len(text) > self._max_chars for text in texts)
        if chunked:
            logger.warning(
                "GigaEmbed: %d из %d текстов длиннее %d символов — разбиты на части",
                chunked,
                len(texts),
                self._max_chars,
            )

        chunks = []
        spans = []
        for text in texts:
            start = len(chunks)
            parts = [
                text[offset : offset + self._max_chars]
                for offset in range(0, len(text), self._max_chars)
            ]
            chunks.extend(parts or [""])
            spans.append((start, len(chunks)))

        all_embeddings: List[List[float]] = []
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start:start + self._batch_size]
            all_embeddings.extend(self._embed_batch(batch))
        if len(all_embeddings) != len(chunks):
            raise RuntimeError(
                f"GigaEmbed: получено {len(all_embeddings)} embeddings для {len(chunks)} частей"
            )
        vectors = np.asarray(all_embeddings, dtype=float)
        return np.asarray(
            [vectors[start:end].mean(axis=0) for start, end in spans],
            dtype=float,
        )

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        """Эмбеддинг одного батча.

        - Сетевые ошибки → повтор с backoff (до self._retries раз).
        - Лимит токенов (413) → делим тексты и усредняем embeddings частей.
        """
        last_err: Optional[Exception] = None
        net_attempts = 0
        while True:
            try:
                response = self.embeddings(batch)
                return [item.embedding for item in response.data]
            except Exception as err:
                last_err = err
                if self._is_token_limit_error(err) and any(
                    len(text) > self._min_chars for text in batch
                ):
                    logger.warning(
                        "GigaEmbed: лимит токенов — разбиваю %d текстов на части",
                        len(batch),
                    )
                    pooled = []
                    for text in batch:
                        if len(text) <= self._min_chars:
                            pooled.extend(self._embed_batch([text]))
                            continue
                        middle = len(text) // 2
                        vectors = self._embed_batch([text[:middle], text[middle:]])
                        pooled.append(np.asarray(vectors, dtype=float).mean(axis=0).tolist())
                    return pooled
                net_attempts += 1
                if net_attempts >= self._retries:
                    raise RuntimeError(
                        f"GigaEmbed: исчерпаны попытки ({self._retries})"
                    ) from last_err
                wait = self._base_backoff * (2 ** (net_attempts - 1))
                logger.warning(
                    "GigaEmbed batch failed (attempt %d/%d): %s; повтор через %.1fs",
                    net_attempts, self._retries, err, wait,
                )
                time.sleep(wait)
