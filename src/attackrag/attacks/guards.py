from __future__ import annotations

# Back-compat shim: логика защит переехала в `attackrag.defenses`.
from attackrag.defenses.guards import DataFilter, InputFilter, OutputVerifier

__all__ = ["InputFilter", "DataFilter", "OutputVerifier"]

