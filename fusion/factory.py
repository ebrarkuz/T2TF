"""Konfigurasyondan fuzyon algoritmasi olusturan fabrika."""

from __future__ import annotations

from typing import Any

from .adapters import AdvancedCAAdapter, BasicCVAdapter, DualIMMAdapter
from .base import FusionAlgorithm


_ALGORITHMS = {
    "basic_cv": BasicCVAdapter,
    "advanced_ca": AdvancedCAAdapter,
    "dual_imm": DualIMMAdapter,
}


def available_algorithms() -> tuple[str, ...]:
    return tuple(_ALGORITHMS)


def create_fusion_algorithm(
    algorithm_name: str,
    parameters: dict[str, Any] | None = None,
) -> FusionAlgorithm:
    name = str(algorithm_name).strip().lower()
    try:
        adapter = _ALGORITHMS[name]
    except KeyError as exc:
        raise ValueError(
            f"Bilinmeyen fuzyon algoritmasi: {algorithm_name!r}. "
            f"Secenekler: {', '.join(available_algorithms())}"
        ) from exc
    return adapter(dict(parameters or {}))
