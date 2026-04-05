"""Пакет стенда AttackRAG (одно имя с репозиторием).

Сейчас: baseline RAG + RAGAS + CLI. Атаки/защиты — в подпакетах `attacks`, `defenses` (пока заглушки).

Схема каталогов: ARCHITECTURE.md в корне репозитория.
"""

from attackrag.rag import RAGConfig, RAGPipeline

__all__ = ["RAGConfig", "RAGPipeline"]
