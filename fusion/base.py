"""Fuzyon cekirdekleri icin ortak, runtime'dan bagimsiz arayuz."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FusionAlgorithm(Protocol):
    """Bir radar olcumunu isleyip normalize track snapshot'lari dondurur."""

    def process_measurement(self, measurement: Any) -> list[dict[str, Any]]:
        ...

    def reset(self) -> None:
        ...

    def get_diagnostics(self) -> dict[str, Any]:
        ...
