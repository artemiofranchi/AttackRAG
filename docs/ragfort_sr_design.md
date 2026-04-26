# RAGFort-SR — архитектурный дизайн реализации

> Документ фиксирует план кода **до** правок. Соответствует
> математической модели из `docs/formalization_sr_draft.md` (§2.3.4).

## 1. Состав модулей

| Модуль | Файл | Что делает | Зависимости |
|---|---|---|---|
| Contrastive Re-indexing (упрощ.) | `defenses/ragfort_sr/reindex.py` | Hard-negative mining между соседними K-means кластерами: пересчитывает эмбеддинги чанков так, чтобы внутрикластерные были ближе, межкластерные дальше — через линейную проекцию, обученную на triplet-loss’е. Работает off-line. | numpy, sklearn, E |
| Constrained Cascade Generation | `defenses/ragfort_sr/cascade.py` | Draft-then-Verify: сначала генерируем черновик, потом один из верификаторов из пула принимает решение. Использует `RandomizedVerifier`. | LLMClient |
| Centroid Proximity Guard (**CPG**) | `defenses/ragfort_sr/cpg.py` | Вычисляет `ρ(q)`, сравнивает с порогом `τ_ρ`, возвращает `bool`. | E, центроиды |
| Coverage Budget (**CB**) | `defenses/ragfort_sr/coverage.py` | Учёт `Cov(u, Δt)` per-session, три режима (pass/strict/block). | время, session_id |
| Decomposition Detector (**DD**) | `defenses/ragfort_sr/decomposition.py` | Regex-словари + LLM-классификатор EDEA-композиции. | LLMClient |
| Randomized Verifier (**RV**) | `defenses/ragfort_sr/verifier.py` | Пул из K промпт-вариантов верификатора, выбор по `π`. | LLMClient |
| Оркестратор | `defenses/ragfort_sr/pipeline.py` | Последовательно собирает все компоненты в `DefendedRAGPipeline`. | всё вышестоящее + RAGPipeline |

Публичная точка входа: `from attackrag.defenses.ragfort_sr import RAGFortSR`.
Старые файлы `defenses/ragfort.py`, `defenses/leaksealer.py`,
`defenses/controlnet.py` **сохраняем** — они нужны для сравнительных
таблиц (RAGFort baseline, LeakSealer baseline, ControlNET baseline).

## 2. Контракты

### 2.1 CPG
```python
class CentroidProximityGuard:
    def fit(self, *, embedder: EmbeddingModel, chunk_texts: list[str], n_clusters: int): ...
    def rho(self, query: str) -> float: ...
    def is_attack(self, query: str) -> bool: ...   # rho(query) < tau_rho
    def calibrate(self, benign_queries: list[str], fpr_target: float) -> None: ...
```
Порог `τ_ρ` калибруется один раз по quantile'у `1 - fpr_target` над
легитимными запросами.

### 2.2 CB
```python
class CoverageBudget:
    def __init__(self, *, beta_cov: int, beta_max: int, window_seconds: float): ...
    def observe(self, session_id: str, cluster_id: int, ts: float) -> Literal["pass","strict","block"]: ...
```
Внутри — OrderedDict по сессиям с TTL. В реальной системе — Redis,
для стенда — in-memory.

### 2.3 DD
```python
class DecompositionDetector:
    ext_lexicon: tuple[str, ...]
    jail_lexicon: tuple[str, ...]
    llm: LLMClient | None          # опционально, для LLM-подтверждения
    def is_attack(self, query: str, rho: float | None = None) -> bool: ...
```
Булев AND: `φ_ext ∧ φ_jail ∧ φ_trg`. Для `φ_trg` — `rho < τ'_ρ` или
максимальная cos-сходимость чанка > `τ_trg`.

### 2.4 RV
```python
@dataclass
class RandomizedVerifier:
    prompts: list[str]                   # K вариантов
    temperatures: list[float]
    seeds: list[int]
    llm: LLMClient
    def verify(self, question: str, contexts: list[str], draft: str, rng: np.random.Generator) -> bool: ...
```
`rng` прокидывается из оркестратора, чтобы атака не могла
детерминировать выбор. В strict-режиме CB делает 2 независимые
прогонки с AND-логикой.

### 2.5 Оркестратор
```python
@dataclass
class RAGFortSR:
    cpg: CentroidProximityGuard
    cb: CoverageBudget | None
    dd: DecompositionDetector
    cascade: ConstrainedCascade       # использует RV внутри
    reindex: ContrastiveReindex | None

    def answer(self, question: str, *, session_id: str | None = None) -> DefendedAnswer: ...
```
`DefendedAnswer` возвращает не только ответ, но и подробный trace
(какой модуль сработал, значения ρ/Cov/vote V) — это нам нужно для
отчётных табличек и BPD-замеров.

## 3. Границы ответственности

- `CPG` и `DD` **только** решают, идти ли дальше. Они не модифицируют
  запрос, не фильтруют контекст.
- `CRI` (Contrastive Re-indexing) переписывает индекс **offline** через
  CLI-команду `attack-rag-reindex`. В runtime только подгружаем
  переиндексированную версию.
- `CB` — stateful; для детерминированности тестов есть режим
  `--cb-disabled`.
- `RV` — **единственный** компонент, который может «блокировать»
  ответ (возвращает `Content blocked`), наследуя интерфейс исходного
  RAGFort-верификатора.

## 4. План реализации (последовательность PR-ов)

1. Скелет пакета `defenses/ragfort_sr/` + типы + `DefendedAnswer`.
2. `CPG` + калибровка + юнит-тесты на синтетических эмбеддингах.
3. `DD` + словари + юнит-тесты на фикстурах безобидных и EDEA-запросов.
4. `RV` поверх существующего `OutputVerifier` + пул из 3–5 промптов.
5. `CB` in-memory + тесты границ.
6. `CRI`-stub: упрощённая переиндексация (triplet-loss поверх
   линейной проекции, обучаемой 1 эпоху на GPU/CPU ~15 минут).
7. Оркестратор `RAGFortSR` + интеграция в `run_attack_suite`.
8. Переработка `attacks/secret_lite.py` → настоящий SECRET
   (LLM-as-Optimizer + полноценный CFT с двумя фазами).

## 5. Соответствие формализации

| Формула §2.3.4 | Реализация |
|---|---|
| `ρ(q) = min_i ‖E(q)-μ_i‖ / median_j …` | `CPG.rho()` |
| `Cov(u, Δt)` | `CB.observe()` + словарь сессий |
| `φ_ext ∧ φ_jail ∧ φ_trg` | `DD.is_attack()` |
| `E_π[P(Miss_V)]` | `RV.verify()` с ансамблем |
| `γ = Π(1 - TPR_*)` | считается в `run_attack_suite` после прогона |

## 6. Что остаётся на этап 2 (код после формализации)

- Настоящий SECRET-оптимизатор для атак (`attacks/secret_full.py`) —
  чтобы RV имел против кого защищаться, иначе adaptive-тест выродится.
- Калибровочный CLI `attack-rag-calibrate-defenses` для `τ_ρ`, пулов
  промптов, `β_cov`.
- Бюджет экспериментов (пункт 3 запроса пользователя) — отдельный
  документ `experiments_budget.md`.
