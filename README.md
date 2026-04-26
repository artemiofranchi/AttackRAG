# AttackRAG

**Программный комплекс по оценке и защите RAG-систем от атак извлечения данных.**

Магистерская диссертация: разработка ПК, который собирает RAG из взаимозаменяемых компонентов, измеряет его базовое качество, проводит атаки извлечения (Prompt Injection, Backdoor, SECRET) и сравнивает защитные профили — `none`, `basic-filters`, `ragfort` (полная black-box-адаптация RAGFort), `hard` (предлагаемый метод HARD). Ключевые формулировки и параметры эксперимента — в `docs/thesis_formalization.tex`. План реализации — в `docs/agent_prompt_implementation.md`. Архитектура — в `ARCHITECTURE.md`.

> Состояние: продолжается интеграция шагов 0–10 плана `docs/agent_prompt_implementation.md`. Команды ниже соответствуют целевому состоянию ПК; некоторые из них появятся по мере выполнения соответствующих шагов плана.

## Возможности

- Собирает RAG-систему на корпусе из `.md`-файлов с настраиваемыми компонентами:
  - векторное хранилище: **Qdrant** (server через `docker compose` или embedded), legacy: numpy/FAISS/Chroma;
  - эмбеддер: любой `sentence-transformers`-совместимый;
  - LLM по ролям (`GENERATOR`, `VERIFIER`, `JUDGE`, `SEGMENTER`): локально через Ollama (Llama 3.1 8B) или по API (Gemini 2.0 Flash через OpenAI-совместимый endpoint).
- Проводит атаки извлечения данных:
  - **Prompt Injection** — 15–20 шаблонов, multi-lingual, markup-rendering;
  - **Backdoor (data poisoning)** — отравление корпуса через стандартный канал ингеста до индексации;
  - **SECRET** — LLM-as-Optimizer для $O_{jail}$ и Cluster-Focused Triggering для $T_{retr}$.
- Применяет защитные профили: `none`, `basic-filters`, `ragfort`, `hard`.
- Считает метрики: ASR, FPR, BPD, $\Delta T$, StagePassRate, $\mathrm{AUC}_{\text{IRD}}$, $\kappa$ (см. §2.4 формализации).
- Запускается из CLI (Typer) и Web-UI (Streamlit).

## Быстрый старт (Python 3.12)

### 1. Установка пакета

```powershell
cd C:\Users\vad\ДипломRAG_Extraction\AttackRAG
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.12 -m pip install -e .
copy .env.example .env
```

На macOS/Linux эквивалентно:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python3.12 -m pip install -e .
cp .env.example .env
```

### 2. Конфигурация `.env`

Минимально нужны:

```dotenv
# Векторное хранилище (по умолчанию qdrant)
VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333          # пусто => embedded режим в data/qdrant_storage/

# Провайдеры LLM по ролям
LLM_GENERATOR_PROVIDER=ollama
LLM_GENERATOR_MODEL=llama3.1:8b-instruct-q4_K_M
OLLAMA_BASE_URL=http://localhost:11434

LLM_VERIFIER_PROVIDER=openai_compat
LLM_VERIFIER_MODEL=gemini-2.0-flash
LLM_VERIFIER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_VERIFIER_API_KEY=...   # ключ Gemini

LLM_JUDGE_PROVIDER=openai_compat
LLM_JUDGE_MODEL=gemini-2.0-flash
LLM_JUDGE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_JUDGE_API_KEY=...

LLM_SEGMENTER_PROVIDER=openai_compat
LLM_SEGMENTER_MODEL=gemini-2.0-flash
LLM_SEGMENTER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_SEGMENTER_API_KEY=...

LLM_CACHE_DIR=runs/llm_cache
```

Полный шаблон — в `.env.example`.

### 3. Qdrant в Docker (рекомендуется)

```bash
docker compose up -d qdrant
# Web UI: http://localhost:6333/dashboard
```

Альтернатива — embedded-режим: оставьте `QDRANT_URL` пустым, Qdrant создаст хранилище в `data/qdrant_storage/`.

### 4. Корпус и индекс

```bash
attack-rag-fetch-wikipedia --max-docs 300
mkdir -p data/corpus_merged
cp data/corpus/*.md data/corpus_merged/
cp data/corpus_wikipedia/*.md data/corpus_merged/

attack-rag-build-index --corpus data/corpus_merged --backend qdrant
```

Для Backdoor:

```bash
attack-rag-build-poison-index --corpus data/corpus_merged --out data/index_poisoned --poison-rate 0.005
```

### 5. Baseline-качество RAG

```bash
attack-rag-benchmark-baseline --questions data/golden_qa.json
```

### 6. Главный эксперимент

Повседневный прогон (один сид):

```bash
attack-rag-experiment \
  --defenses none basic-filters ragfort hard \
  --attacks pi backdoor secret \
  --seeds 42 \
  --out runs/experiment_$(date +%s)
```

Быстрая проверка стенда (≈5 минут):

```bash
attack-rag-experiment --smoke --out runs/smoke
```

Финальный прогон для диссертации (3–4 часа, mean ± std):

```bash
attack-rag-experiment --defenses none basic-filters ragfort hard \
                      --attacks pi backdoor secret \
                      --seeds 41 42 43 \
                      --out runs/final
```

Режимы прогона по времени:

| Флаг        | Назначение                                | Сидов | Вопросов / атаку | SECRET-итераций | Время на M3 8/512 |
|-------------|-------------------------------------------|-------|------------------|-----------------|-------------------|
| `--smoke`   | дымовой тест UI/CLI                        | 1     | 5–10             | 1               | ≈ 5 мин           |
| `--fast`    | отладка стенда, ablation                   | 1     | 50               | 3               | 15–25 мин         |
| (без флагов)| обычный прогон с одним сидом               | 1     | 200              | 5               | 60–90 мин         |
| `--seeds 41 42 43` | финальный прогон для диссертации   | 3     | 200              | 5               | 3–4 ч             |

### 7. Web-UI (Streamlit)

```bash
streamlit run apps/streamlit_app.py
```

В UI выбираются: корпус, бэкенд индекса (qdrant/embedded), модели по ролям, защитный профиль, набор атак, режим прогона (smoke/fast/normal). Кнопка «Run experiment» запускает `attack-rag-experiment` через `subprocess.run`, прогресс и логи отображаются в реальном времени, после завершения показывается сводная таблица метрик и доступны ссылки на скачивание `report.json`, `report.csv`, графиков из `runs/<ts>/`.

UI и CLI используют один и тот же `Experiment Orchestrator` — результаты идентичны (NFR-6).

### 8. Графики для диссертации

```bash
attack-rag-plot-experiment --run runs/<ts>
```

Создаёт в `runs/<ts>/figures/`: bar-plots ASR/BPD/Latency, ROC IRD, Stage Pass Rates, Cascade Bound Tightness $\kappa$.

## Данные

- `data/corpus/` — небольшой вымышленный корпус (в т.ч. «конфиденциальные» файлы для атак извлечения).
- `data/corpus_wikipedia/` — статьи Wikipedia из датасета [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (**CC BY-SA**); в `.gitignore`.
- `data/corpus_merged/` — объединённый корпус для индексации.
- `data/golden_qa.json` — 30 пар (вопрос, эталонный ответ) для baseline-качества и BPD.
- `data/benign_training.json` — ~50 «фоновых» вопросов для калибровки порога $\theta_1$ IRD.
- `data/leak_patterns.yaml` — regex-паттерны для Output Leak Scanner.
- `data/index/`, `data/index_poisoned/`, `data/qdrant_storage/`, `runs/` — артефакты, в `.gitignore`.

## Документация

- [`docs/thesis_formalization.tex`](docs/thesis_formalization.tex) — формализация, требования (FR-1..FR-10, NFR-1..NFR-7), параметры эксперимента (источник истины).
- [`docs/agent_prompt_implementation.md`](docs/agent_prompt_implementation.md) — пошаговый план реализации (контракт для AI-агента).
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — архитектура программного комплекса.

## Ограничения и допущения

- Модель угроз — **строго black-box**: атакующий имеет доступ только к обычному пользовательскому интерфейсу RAG-системы, не имеет доступа к скрытым активациям LLM, к векторному индексу напрямую и к весам моделей.
- Backdoor моделируется отравлением корпуса до индексации через любой стандартный канал ингеста (открытый источник, пользовательская загрузка, коннектор) — комплекс не предъявляет требований к конкретному каналу.
- Метод HARD строится поверх полной black-box-адаптации RAGFort (Draft-then-Verify + Topic-Consistent Re-ranking), оригинальный вклад работы — компоненты IRD, Output Leak Scanner и Risk-Budget Thresholding (см. §2.3.4 формализации).
