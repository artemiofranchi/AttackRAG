from __future__ import annotations

from dataclasses import dataclass

from attackrag.defenses.guards import DataFilter, OutputVerifier


@dataclass
class RAGFortProxyDefense:
    """RAGFort-style proxy: context isolation + constrained cascade generation."""

    verifier: OutputVerifier
    data_filter: DataFilter
    robust_prefix: str

    def robust_question(self, question: str) -> str:
        return f"{self.robust_prefix}\n\n{question}"

    def filter_contexts(self, contexts: list[str]) -> list[str]:
        return self.data_filter.clean(contexts)

    def verify(self, question: str, contexts: list[str], draft: str) -> str:
        return self.verifier.verify(question, contexts, draft)

