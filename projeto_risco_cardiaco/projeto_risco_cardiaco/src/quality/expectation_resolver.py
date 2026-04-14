"""
src/quality/expectation_resolver.py — Resolução dinâmica de classes GE a partir do YAML.
"""
from __future__ import annotations
from src.quality.base import ExpectationResolver

class GeExpectationResolver(ExpectationResolver):
    def __init__(self, gxe) -> None:
        self._gxe = gxe

    def resolve(self, type_name: str) -> type:
        pascal = self._para_pascal(type_name)

        if hasattr(self._gxe, pascal):
            return getattr(self._gxe, pascal)

        if hasattr(self._gxe, type_name):
            return getattr(self._gxe, type_name)

        raise AttributeError(
            f"Expectation '{type_name}' não encontrada no módulo great_expectations.expectations.\n"
            f"Verifique o nome no quality.yaml — use snake_case ou PascalCase exato."
        )

    @staticmethod
    def _para_pascal(snake: str) -> str:
        return "".join(palavra.capitalize() for palavra in snake.split("_"))