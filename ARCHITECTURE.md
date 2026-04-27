# Архитектура программного комплекса AttackRAG

> Документ описывает архитектуру **программного комплекса по оценке и защите RAG-систем от атак извлечения данных** (магистерская диссертация, [Волков А.Д.]). Источник истины по требованиям, формализации и параметрам — `docs/thesis_formalization.tex`, особенно глава 3 (`sec:software_complex`). Любые расхождения между этим файлом и `.tex` решаются в пользу `.tex`.

## Назначение и место в работе

Программный комплекс — один из двух ключевых результатов диссертации (наряду с предложенным методом защиты HARD). Комплекс позволяет:

1. собрать RAG-систему на заданном корпусе из взаимозаменяемых компонентов (векторное хранилище, эмбеддер, LLM по ролям);
2. измерить базовое качество RAG;
3. смоделировать атаки извлечения данных (Prompt Injection, Backdoor, SECRET);
4. подключить защитные профили (`none`, `basic-filters`, `ragfort`, `hard`) и сравнить их;
5. собрать метрики (ASR, FPR, BPD, $\Delta T$, StagePassRate, $\mathrm{AUC}_{\text{IRD}}$, $\kappa$) и графические артефакты для главы экспериментальных результатов;
6. дать пользователю UI и CLI, с которыми эксперимент проводится без правки исходного кода.

## Слойная архитектура

Комплекс состоит из четырёх слоёв (`sec:software_architecture` формализации):

```
┌─────────────────────────────────────────────────────────────┐
│  Presentation Layer    Web UI (Streamlit) │ CLI (Typer)     │
├─────────────────────────────────────────────────────────────┤
│  Pipeline Layer        Attacks │ RAG Pipeline │ Defenses    │
│                              ↘     ↓     ↙                  │
│                            Experiment Orchestrator          │
├─────────────────────────────────────────────────────────────┤
│  Reporting Layer       Metrics Engine │ Reports │ Plots     │
├─────────────────────────────────────────────────────────────┤
│  Data Layer            Corpus Manager │ Index Builder │     │
│                        Poisoned-Index Builder               │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
   External LLM API (black-box, см. sec:threat_model)
```

`Presentation Layer` — тонкая надстройка над `Pipeline Layer` и `Reporting Layer`: UI и CLI вызывают одни и те же функции `Experiment Orchestrator`. Это обеспечивает идентичность результатов между запусками из UI и из командной строки и не вводит отдельного «UI-only»-кода (NFR-6).

## Структура пакета

```text
.
├── apps/
│   └── streamlit_app.py          # Web-UI (FR-10): выбор корпуса, бэкенда, LLM по ролям, профиля, атаки
├── docker-compose.yml            # сервис Qdrant (qdrant/qdrant), порт 6333, том data/qdrant_storage/
├── docs/
│   ├── thesis_formalization.tex  # источник истины: формализация, требования, параметры
│   └── agent_prompt_implementation.md  # план реализации (контракт для AI-агента)
├── data/
│   ├── corpus/                   # базовый вымышленный корпус (.md)
│   ├── corpus_wikipedia/         # 200–500 статей Wikipedia (в .gitignore)
│   ├── corpus_merged/            # объединённый корпус для индексации
│   ├── golden_qa.json            # 36 пар (вопрос, эталонный ответ) для baseline-качества и ASR
│   ├── benign_training.json      # ~50 «фоновых» вопросов для калибровки θ_1 IRD
│   ├── leak_patterns.yaml        # regex-паттерны для Output Leak Scanner
│   ├── index/                    # чистый Qdrant-индекс (в .gitignore)
│   ├── index_poisoned/           # poisoned-индекс для Backdoor (в .gitignore)
│   └── qdrant_storage/           # данные Qdrant в server-режиме (в .gitignore)
├── runs/                         # артефакты прогонов (в .gitignore)
│   ├── llm_cache/                # on-disk кэш LLM-вызовов
│   └── experiment_<ts>/          # результаты единичного прогона
└── src/attackrag/
    ├── __init__.py               # публичный вход: RAGConfig, RAGPipeline, build_pipeline_from_disk
    ├── paths.py                  # пути к data/
    ├── documents.py              # чтение и нормализация .md
    ├── chunking.py               # разбиение на чанки (D)
    ├── embeddings.py             # эмбеддер (E): sentence-transformers
    ├── llm.py                    # LLMClient: OllamaLLM, OpenAICompatLLM (GENERATOR/VERIFIER/JUDGE/SEGMENTER)
    ├── rag.py                    # шаблон промпта T + сборка RAG-pipeline
    ├── ragas_eval.py             # baseline-качество через RAGAS (legacy для шага 1, не для главного эксперимента)
    ├── vector_stores/
    │   ├── loader.py             # build_vector_store(...): default = qdrant
    │   ├── qdrant_store.py       # основное хранилище: server-режим или embedded
    │   ├── numpy_store.py        # legacy fallback (для load старых индексов)
    │   ├── faiss_store.py        # legacy fallback
    │   └── chroma_store.py       # legacy fallback
    ├── attacks/
    │   ├── prompt_injection.py   # PI: 15–20 шаблонов, multi-lingual, markup-rendering
    │   ├── backdoor_attack.py    # Backdoor: триггер τ + payload π
    │   ├── corpus_poison.py      # подготовка poisoned-корпуса
    │   ├── secret.py             # SECRET: LLM-as-Optimizer для O_jail + Cluster-Focused Triggering для T_retr
    │   ├── detectors.py          # LeakDetector: substring / regex / LLM-judge режимы
    │   ├── metrics.py            # ASR, FPR, BPD, latency, stage_pass_rates, stage_correlation, auc_ird, kappa
    │   └── types.py              # Trial, RetrievedChunk, общие dataclass'ы
    ├── defenses/
    │   ├── __init__.py
    │   ├── guards.py             # InputFilter, DataFilter, OutputVerifier (профиль basic-filters)
    │   ├── ird.py                # Intent–Retrieval Dissociation (стадия 1 HARD; eq:ird_score)
    │   ├── tcr.py                # Topic-Consistent Re-ranking (eq:tcr_score; разделяемый между ragfort и hard)
    │   ├── output_scanner.py     # LeakScanner (стадия 4 HARD; eq:leak_scan)
    │   ├── ragfort.py            # RAGFortVerifier (Draft-then-Verify) + RAGFortDefense (DtV + TCR)
    │   ├── cascade.py            # HARDCascade: оркестратор IRD → TCR → DtV → Scanner + Risk-Budget
    │   └── _legacy/              # SOTA-методы, изученные в обзоре, но не реализуемые в эксперименте
    │       ├── __init__.py       # пояснительная заметка
    │       └── leaksealer.py     # LeakSealer (single-query OOD): теоретически слаб против SECRET; см. ниже
    └── cli/
        ├── build_index.py            # attack-rag-build-index --backend qdrant
        ├── build_poisoned_index.py   # attack-rag-build-poison-index
        ├── fetch_wikipedia.py        # attack-rag-fetch-wikipedia
        ├── benchmark_baseline.py     # attack-rag-benchmark-baseline (FR-2)
        ├── run_experiment.py         # attack-rag-experiment (главное CLI; FR-3..FR-9)
        └── plot_experiment.py        # attack-rag-plot-experiment (FR-9)
```

## Соответствие модулей и формализации

| Модуль | Объект формализации | Раздел `.tex` |
|---|---|---|
| `chunking.py`, `documents.py` | $\mathcal{D}$, разбиение на чанки $c_i$ | §2.1 |
| `embeddings.py` | Функция эмбеддинга $E$ | §2.1 |
| `vector_stores/qdrant_store.py` | Векторный индекс $\mathcal{I}$ | §2.1, `tab:tech_stack` |
| `rag.py` | $\mathcal{T}$ (шаблон), $G$ (генератор), `f_RAG = G ∘ T ∘ R` | §2.1 |
| `llm.py` | LLM с разделением по ролям | `tab:llm_roles` |
| `attacks/prompt_injection.py` | PI-атака `q' = q_benign ⊕ δ` | §2.2.1 (`sec:pi_formalization`) |
| `attacks/backdoor_attack.py` + `corpus_poison.py` | Backdoor: $d_{pois}$ через канал ингеста | §2.2 + §2.2.2 (`sec:backdoor_formalization`) |
| `attacks/secret.py` | SECRET: $x = I_{ext} ⊕ O_{jail} ⊕ T_{retr}$ | §2.2.3 (`sec:secret_formalization`) |
| `defenses/guards.py` | Фильтрация (`F_in`, `F_data`), Robust System Prompting | §2.3.2 (`sec:def_basic`) |
| `defenses/ragfort.py` | Полная RAGFort-адаптация: Draft-then-Verify + Topic-Consistent Re-ranking | §2.3.3 (`sec:def_sota`) |
| `defenses/tcr.py` | TCR (разделяемый компонент `ragfort` и `hard`) | §2.3.3, §2.3.4, `eq:tcr_score` |
| `defenses/ird.py` | IRD-сигнал $s_{\text{IRD}}(q)$ (стадия 1 HARD) | §2.3.4, `eq:ird_score` |
| `defenses/output_scanner.py` | Output Leak Scanner (стадия 4 HARD) | §2.3.4, `eq:leak_scan` |
| `defenses/cascade.py` | HARD-каскад + Risk-Budget Thresholding | §2.3.4, `eq:risk_budget`, `tab:hard_stages` |
| `attacks/metrics.py` | Все метрики (ASR…$\kappa$) | §2.4 (`sec:metrics`) |
| `cli/run_experiment.py` | Experiment Orchestrator | §3 (FR-3, FR-4, FR-6, FR-7, FR-9) |
| `apps/streamlit_app.py` | Web UI | §3 (FR-10) |

## Профили защиты

Реализуются через единый `Defense Manager` (см. §3 `tab:defense_profiles` формализации):

| Профиль | Содержимое (модули) | Назначение |
|---|---|---|
| `none` | — | Контроль |
| `basic-filters` | `guards.InputFilter` + `guards.DataFilter` + robust system prompt в `rag.py` | Простые меры (уровень курсовой) |
| `ragfort` | `tcr.TopicConsistentReranker` + `ragfort.RAGFortVerifier` (Draft-then-Verify) | **Главный baseline** для HARD: полная black-box-адаптация RAGFort |
| `hard` | `ragfort` + `ird.IRDDetector` + `output_scanner.LeakScanner` + `cascade.HARDCascade` (Risk-Budget) | Полный предлагаемый метод |

Сравнение `hard` − `ragfort` напрямую измеряет вклад трёх оригинальных компонентов настоящей работы: IRD, Output Scanner, Risk-Budget Thresholding. TCR и Draft-then-Verify — **разделяемые** компоненты RAGFort-адаптации, реализованные в одном экземпляре и одинаково используемые в `ragfort` и `hard`. Это исключает спор «вы превзошли RAGFort, потому что ослабили baseline».

## Роли LLM

Внутри комплекса языковые модели разнесены по четырём ролям. Каждая конфигурируется независимо через `.env` (см. `tab:llm_roles` формализации):

| Роль | Использование | Рекомендуемая модель / провайдер |
|---|---|---|
| `GENERATOR` | RAG-генерация ответа (массовый вызов); Draft в стадии 3 HARD | Llama 3.1 8B Instruct через Ollama (локально, дёшево) либо Gemini 2.0 Flash через API |
| `VERIFIER` | Draft-then-Verify (профиль `ragfort` и стадия 3 HARD) | **Gemini 2.0 Flash** или сильнее, через OpenAI-compat API. Использование локальной Llama здесь запрещено. |
| `JUDGE` | LLM-as-a-judge для $\mathbb{I}_{leak}$ и для оценки качества | **Gemini 2.0 Flash** или сильнее, через API (не должен совпадать с `GENERATOR` для независимости оценки). |
| `SEGMENTER` | Декомпозиция запроса $\phi(q)$ для IRD | Gemini 2.0 Flash через API; для отладки — Llama 3.1 8B через Ollama |

Один API-ключ Gemini может покрывать `VERIFIER`, `JUDGE`, `SEGMENTER`; кэш LLM-вызовов ключуется ролью (NFR-3).

## `_legacy/`: что и почему

Подпакет `defenses/_legacy/` содержит модули, изученные в обзоре SOTA, но **не реализуемые** как самостоятельные защитные профили в эксперименте. Причины — в `docs/thesis_formalization.tex` §2.3.3 (ремарки `rem:leaksealer_limit`, `rem:controlnet_limit`):

- `leaksealer.py` (LeakSealer, [Panebianco 2025]) — single-query OOD-сигнал теоретически слаб против SECRET-атак (триггер целенаправленно оптимизирован на близость к центроиду легитимного кластера, см. `eq:cft`). Идея кластерного анализа эмбеддингов **переиспользована в `defenses/ird.py`**, но применяется там не к запросу целиком, а к ретривальным отпечаткам его семантических фрагментов — это и есть IRD.
- `controlnet.py` (ControlNET, [Yao 2025]) — **удалён полностью**, не перенесён в `_legacy/`. Метод опирается на анализ скрытых активаций LLM, что в принятой black-box-модели угроз недоступно. Эмбеддинговый суррогат, ранее лежавший в этом файле, по форме совпадал с LeakSealer и не нёс независимой информации, нарушая допущение условной независимости сигналов в теореме каскадной границы (`thm:cascade_bound`).

Модули из `_legacy/` не импортируются в `cascade.py`, в основной `defenses/__init__.py` и в `cli/run_experiment.py`.

## Поток данных

```mermaid
flowchart LR
  subgraph data["Data Layer"]
    D[data/corpus_merged]
    PD[data/corpus_poisoned]
    I[data/index]
    PI[data/index_poisoned]
    D --> I
    PD --> PI
  end
  subgraph pipe["Pipeline Layer"]
    A[Attack Suite<br/>PI · Backdoor · SECRET]
    R[RAG Pipeline<br/>q → R(q,I) → G]
    DEF[Defense Manager<br/>none · filters · ragfort · hard]
    O[Experiment Orchestrator]
    A --> R
    DEF --> R
    O --> A
    O --> DEF
    O --> R
  end
  subgraph rep["Reporting Layer"]
    M[Metrics Engine]
    REP[Report Builder]
    P[Plot Generator]
    O --> M --> REP --> P
  end
  subgraph pres["Presentation Layer"]
    UI[Web UI · Streamlit]
    CLI[CLI · attack-rag-experiment]
    UI --> O
    CLI --> O
  end
  I --> R
  PI --> R
  R -.black-box.-> LLM[(External LLM API)]
  DEF -.black-box.-> LLM
```

## Конфигурация и воспроизводимость

- `.env` — провайдеры и модели по ролям, ключи API, `QDRANT_URL`, `LLM_CACHE_DIR`, `RAG_TOP_K`, `RAG_CANDIDATE_K`. Шаблон — в `.env.example`.
- Кэш LLM — on-disk JSON в `runs/llm_cache/` (NFR-3). Ключ: SHA256(provider + model + role + prompt + temperature + max_tokens).
- `provenance.json` для каждого прогона: git-коммит, версия корпуса (хеш), параметры индекса, имена/версии моделей по ролям, сиды, активный профиль, env-переменные без секретов.
- Сиды: по умолчанию `{42}` (один сид) — для повседневной работы. Для финального прогона: `--seeds 41,42,43` (`mean ± std`).
- Режимы CLI: `--smoke` (5 минут), `--fast` (15–25 минут), без флагов (60–90 минут), `--seeds 41,42,43` (3–4 часа).

## Что НЕ входит в архитектуру

- Никаких white-box защит (требующих доступа к активациям/логитам LLM). Модель угроз (`sec:threat_model`) — строго black-box.
- Никакого обучения собственных эмбеддеров (Contrastive Re-indexing из RAGFort заменён на TCR на фиксированном эмбеддере).
- Никакого token-level Constrained Cascade Generation (заменён на response-level Draft-then-Verify).
- Никакой защиты от прямой компрометации векторного индекса, эмбеддера или весов LLM — это инфраструктурная угроза вне модели.
- Numpy/FAISS/Chroma как дефолтные backend'ы — оставлены только для load-совместимости со старыми индексами; новые индексы строятся на Qdrant.

## Дополнительные документы

- `docs/thesis_formalization.tex` — формализация, требования, параметры эксперимента (источник истины).
- `docs/agent_prompt_implementation.md` — пошаговый план реализации (контракт для AI-агента).
- `README.md` — быстрый старт, команды, переменные окружения.
