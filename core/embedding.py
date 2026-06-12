from __future__ import annotations

import hashlib
from typing import Optional

from core.config import AppConfig, load_config


def embed_text(
    text: str, config: Optional[AppConfig] = None, dimensions: int = 8
) -> list[float]:
    _ = config or load_config()
    safe_dimensions = max(1, dimensions)
    seed = text.encode("utf-8")
    values: list[float] = []
    counter = 0

    while len(values) < safe_dimensions:
        digest = hashlib.sha256(seed + counter.to_bytes(2, "big")).digest()
        for byte in digest:
            values.append(round((byte / 255.0) * 2 - 1, 6))
            if len(values) == safe_dimensions:
                break
        counter += 1

    return values
