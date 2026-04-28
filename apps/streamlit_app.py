"""
AttackRAG — Web-UI программного комплекса (FR-10).

Три смысловых блока на одной странице (Configure / Run / Results).

* Configure   — что прогонять и на каких моделях.
* Run         — рендерит точную CLI-команду и стримит её stdout.
* Results     — таблицы (mean ± std, per-(seed × profile × attack)) + zip.

Все «тонкие» гиперпараметры спрятаны в свернутые блоки `st.expander(...)`.
По умолчанию подставляем значения из `.env` пользователя; флаг можно
переключить на Gemini в любом role-form'е.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

from attackrag.paths import default_golden_qa_path, default_index_dir, default_poisoned_index_dir  # noqa: E402

load_dotenv(_REPO / ".env")

PROFILE_OPTIONS = ["none", "basic-filters", "ragfort", "hard"]
ATTACK_OPTIONS = ["pi", "backdoor", "secret"]
MODE_OPTIONS = ["smoke", "fast", "default"]
VECTOR_BACKENDS = ["qdrant", "numpy", "faiss", "chroma"]

# Популярные эмбеддинги (для индекса; реально используется только та, на
# которой собран индекс — мы не можем поменять её в рантайме).
EMBEDDING_PRESETS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
]

# Реранкеры (BGE-серия + опция «выкл»).
RERANKER_PRESETS = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-large",
    "BAAI/bge-reranker-base",
]


# ============================================================================
# Хелперы
# ============================================================================
def _list_corpora() -> list[str]:
    out: list[str] = []
    base = _REPO / "data"
    if base.is_dir():
        for p in sorted(base.iterdir()):
            if p.is_dir() and p.name.startswith("corpus"):
                out.append(str(p))
    return out


def _list_runs() -> list[Path]:
    runs = _REPO / "runs"
    if not runs.is_dir():
        return []
    return sorted(
        (p for p in runs.iterdir() if p.is_dir() and p.name.startswith("experiment_")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _read_index_meta(index_dir: Path) -> dict[str, Any]:
    """Читаем `index_meta.json` (если есть), чтобы показать модель эмбеддингов индекса."""
    p = index_dir / "index_meta.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _count_chunks(index_dir: Path) -> int:
    """Сколько чанков в индексе — для отображения статуса. Читаем chunks.json."""
    p = index_dir / "chunks.json"
    if not p.is_file():
        return 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else 0
    except json.JSONDecodeError:
        return 0


def _run_build(cmd: list[str], *, qdrant_url: str, label: str) -> None:
    """Запускает сборку индекса в подпроцессе и стримит stdout в Streamlit.

    Если задан `qdrant_url` — пробрасывает его в env, чтобы build шёл в Docker
    Qdrant (а не embedded — иначе словим тот же AlreadyLocked-баг).
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO / "src")
    if qdrant_url.strip():
        env["QDRANT_URL"] = qdrant_url.strip()
    log_box = st.empty()
    progress = st.empty()
    progress.info(f"Сборка {label}-индекса…", icon="⏳")
    buffer: list[str] = []
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            buffer.append(line.rstrip())
            if len(buffer) > 300:
                buffer = buffer[-300:]
            log_box.code("\n".join(buffer[-200:]))
        rc = proc.wait()
    except Exception as e:  # noqa: BLE001 — показать пользователю
        progress.error(f"Не удалось собрать {label}-индекс: {e}")
        return
    if rc == 0:
        progress.success(f"{label}-индекс собран за {time.time() - t0:.0f}s.", icon="✅")
        st.rerun()
    else:
        progress.error(f"{label}-индекс не собран (exit code {rc}). См. лог выше.")


def _zip_directory(d: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in d.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(d))
    return buf.getvalue()


# ============================================================================
# LLM role form (одна общая компонента для GENERATOR / VERIFIER / JUDGE / SEGMENTER)
# ============================================================================
def _llm_role_form(
    label: str,
    *,
    default_provider: str,
    default_model: str,
    default_base_url: str = "",
    api_key_env: str | None = None,
    help_text: str = "",
) -> dict[str, str]:
    with st.expander(f"LLM: {label}", expanded=False):
        if help_text:
            st.caption(help_text)
        provider = st.selectbox(
            f"{label} provider",
            options=["ollama", "openai_compat (Gemini / OpenAI)"],
            index=0 if default_provider == "ollama" else 1,
            key=f"{label}_provider",
        )
        is_ollama = provider == "ollama"
        model = st.text_input(
            f"{label} model",
            value=default_model,
            key=f"{label}_model",
            help=(
                "Имя модели. Для Ollama — как в `ollama list` (например `gpt-oss:20b-cloud`). "
                "Для openai_compat — `gemini-2.0-flash` / `gpt-4o-mini`."
            ),
        )
        base_url = st.text_input(
            f"{label} base_url",
            value=(default_base_url if not is_ollama else os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")),
            key=f"{label}_base_url",
            help=("Для Ollama — `OLLAMA_HOST`; для Gemini — OpenAI-совместимый эндпоинт."),
        )
        api_key_default = os.environ.get(api_key_env) if (api_key_env and not is_ollama) else ""
        api_key = st.text_input(
            f"{label} API key",
            value=api_key_default or "",
            type="password",
            key=f"{label}_api_key",
            help=(
                "Для Ollama не нужен. Для Gemini — `GEMINI_API_KEY`. "
                "В provenance.json НЕ записывается."
            ),
        )
    return {
        "provider": "ollama" if is_ollama else "openai_compat",
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


# ============================================================================
# Сборка env для подпроцесса (то, что подхватит attack-rag-experiment)
# ============================================================================
def _set_role_env(env: dict[str, str], role: str, spec: dict[str, str]) -> None:
    """Универсальная установка env-переменных для одной из ролей VERIFIER / SEGMENTER."""
    env[f"LLM_{role}_PROVIDER"] = spec["provider"]
    env[f"LLM_{role}_MODEL"] = spec["model"]
    if spec["base_url"]:
        if spec["provider"] == "ollama":
            env[f"LLM_{role}_OLLAMA_HOST"] = spec["base_url"]
        else:
            env[f"LLM_{role}_BASE_URL"] = spec["base_url"]
    if spec["api_key"] and spec["provider"] != "ollama":
        env[f"LLM_{role}_API_KEY"] = spec["api_key"]


def _build_env(form: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO / "src")

    # GENERATOR
    gen = form["generator"]
    env["LLM_PROVIDER"] = gen["provider"]
    if gen["provider"] == "ollama":
        env["OLLAMA_MODEL"] = gen["model"]
        if gen["base_url"]:
            env["OLLAMA_HOST"] = gen["base_url"]
    else:
        env["LLM_MODEL"] = gen["model"]
        if gen["base_url"]:
            env["OPENAI_BASE_URL"] = gen["base_url"]
        if gen["api_key"]:
            env["OPENAI_API_KEY"] = gen["api_key"]
            # Gemini-API-ключ полезен и как GEMINI_API_KEY (его читают другие роли).
            env.setdefault("GEMINI_API_KEY", gen["api_key"])

    _set_role_env(env, "VERIFIER", form["verifier"])
    _set_role_env(env, "SEGMENTER", form["segmenter"])

    # ATTACK_JUDGE — отдельная схема переменных в коде проекта.
    judge = form["judge"]
    env["ATTACK_JUDGE_PROVIDER"] = judge["provider"]
    env["ATTACK_JUDGE_MODEL"] = judge["model"]
    if judge["provider"] == "ollama":
        if judge["base_url"]:
            env["ATTACK_JUDGE_OLLAMA_HOST"] = judge["base_url"]
    else:
        if judge["api_key"]:
            env.setdefault("GEMINI_API_KEY", judge["api_key"])

    # Vector store + RAG
    if form["qdrant_url"]:
        env["QDRANT_URL"] = form["qdrant_url"]
    env["RAG_TOP_K"] = str(form["rag_top_k"])
    env["RAG_CANDIDATE_K"] = str(form["rag_candidate_k"])
    env["RAG_RERANKER_ENABLED"] = "true" if form["reranker_enabled"] else "false"
    if form["reranker_enabled"]:
        env["RAG_RERANKER_MODEL"] = form["reranker_model"]
        env["RAG_RERANKER_BATCH_SIZE"] = str(form["reranker_batch_size"])

    # Тонкие настройки HARD-каскада (advanced)
    h = form["hard_advanced"]
    if h["enabled"]:
        env["HARD_THETA2"] = str(h["theta2"])
        env["HARD_THETA3"] = str(h["theta3"])
        env["HARD_THETA4"] = str(h["theta4"])
        env["HARD_LAMBDA1"] = str(h["lambda1"])
        env["HARD_LAMBDA2"] = str(h["lambda2"])
        env["HARD_LAMBDA3"] = str(h["lambda3"])
        env["HARD_LAMBDA4"] = str(h["lambda4"])
        env["HARD_IRD_QUANTILE"] = str(h["ird_quantile"])
        env["HARD_IRD_KRETR"] = str(h["ird_kretr"])
        env["HARD_LCSR_THRESHOLD"] = str(h["lcsr_threshold"])
        env["HARD_ROBUST_PREFIX"] = "true" if h["robust_prefix"] else "false"
        env["HARD_ADAPTIVE"] = "true" if h["adaptive"] else "false"

    return env


# ============================================================================
# main
# ============================================================================
def main() -> None:
    st.set_page_config(page_title="AttackRAG — программный комплекс (FR-10)", layout="wide")
    st.title("AttackRAG — программный комплекс")
    st.caption("Web-UI оркестратора экспериментов. Один прогон = один вызов `attack-rag-experiment`.")

    tab_cfg, tab_run, tab_results = st.tabs(["1. Configure", "2. Run", "3. Results"])

    # ========================================================================
    # 1. CONFIGURE
    # ========================================================================
    with tab_cfg:
        col_main, col_models = st.columns([1, 1])

        # ----------------------------------------------------------------
        # Левая колонка — корпус, индекс, vector store, профили/атаки/режим
        # ----------------------------------------------------------------
        with col_main:
            # ============================================================
            # 1.1. КОРПУС
            # ============================================================
            st.subheader("Корпус (.md документы)")
            st.caption(
                "Каталог с Markdown-файлами, по которым строится векторный индекс. "
                "Каждый `.md` режется на чанки (~600 символов), кодируется эмбеддинг-моделью "
                "и кладётся в Qdrant. Без индекса RAG-пайплайн не работает."
            )
            corpora = _list_corpora()
            corpus_dir = st.selectbox(
                "Корпус (.md)",
                options=corpora or [str(_REPO / "data" / "corpus_merged")],
                index=0,
                help="Список каталогов `data/corpus*` обнаруживается автоматически.",
            )
            golden_path = st.text_input(
                "Golden QA (вопросы-эталоны)",
                value=str(default_golden_qa_path()),
                help="JSON со списком {question, ground_truth} — служит для измерения ASR/FPR/BPD.",
            )

            # ============================================================
            # 1.2. ПАРАМЕТРЫ ИНДЕКСА (одни на baseline и poisoned)
            # ============================================================
            st.subheader("Параметры индекса")
            st.caption(
                "Эти параметры зашиваются в индекс при сборке. Поменять их в рантайме нельзя — "
                "только пересобрать индекс."
            )
            col_emb, col_back = st.columns([2, 1])
            with col_emb:
                emb_model = st.selectbox(
                    "Embedding model (HuggingFace)",
                    options=EMBEDDING_PRESETS,
                    index=0,
                    help=(
                        "Multilingual MiniLM — лёгкая (~120 МБ), быстрый старт. "
                        "BGE-M3 / E5-large — качественнее, но жрут больше RAM/диска и индексируют дольше."
                    ),
                )
                emb_model_custom = st.text_input(
                    "…или вручную (HuggingFace ID)",
                    value="",
                    placeholder="например, sentence-transformers/all-MiniLM-L6-v2",
                )
                effective_emb = emb_model_custom.strip() or emb_model
            with col_back:
                backend = st.radio(
                    "Backend",
                    options=VECTOR_BACKENDS,
                    index=0,
                    help="Qdrant — по умолчанию. numpy/faiss/chroma — legacy.",
                )
                skip_wiki = st.checkbox(
                    "skip-wiki",
                    value=True,
                    help=(
                        "Не подмешивать русскую Wikipedia (300 случайных статей) в индекс. "
                        "Включи, если хочешь только свой корпус (быстрее)."
                    ),
                )
            qdrant_url = st.text_input(
                "QDRANT_URL (пусто = embedded в data/index/qdrant_storage/)",
                value=os.environ.get("QDRANT_URL", ""),
                disabled=(backend != "qdrant"),
                help=(
                    "Если поднял `docker compose up -d qdrant` — задай `http://localhost:6333`. "
                    "Embedded режим работает «из коробки», но НЕ поддерживает параллельных клиентов."
                ),
            )
            if backend == "qdrant" and not qdrant_url.strip():
                st.warning(
                    "Выбран embedded Qdrant. Если этот UI работает одновременно с CLI или "
                    "прогоном — упадёт `Storage folder is already accessed`. "
                    "Поднять server: `docker compose up -d qdrant`, потом задать "
                    "`QDRANT_URL=http://localhost:6333`.",
                    icon="⚠️",
                )

            # ============================================================
            # 1.3. BASELINE INDEX (для PI / SECRET / бенигн)
            # ============================================================
            st.subheader("Индекс baseline")
            st.caption(
                "Используется атаками **PI** и **SECRET**, а также для бенигн-вопросов "
                "(чтобы измерить FPR/BPD на легитимных запросах)."
            )
            index_dir = st.text_input("Каталог индекса", value=str(default_index_dir()))
            meta = _read_index_meta(Path(index_dir))
            chunks_n = _count_chunks(Path(index_dir))
            current_emb = meta.get("embedding_model")
            mismatch = current_emb and effective_emb and current_emb != effective_emb

            if not meta:
                st.warning(
                    f"Индекс по пути `{index_dir}` ещё не собран. "
                    "Нажми «Собрать baseline индекс» ниже.",
                    icon="📦",
                )
            else:
                st.success(
                    f"Готов: **{chunks_n}** чанков, эмбеддинги — `{current_emb}`",
                    icon="✅",
                )
                if mismatch:
                    st.warning(
                        f"В индексе модель `{current_emb}`, выбрана `{effective_emb}`. "
                        "Эксперимент будет использовать ту, что зашита в индексе. "
                        "Чтобы применить выбранную — нажми «Пересобрать»."
                    )

            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                build_baseline = st.button(
                    "Собрать baseline индекс" if not meta else "Пересобрать baseline",
                    width="stretch",
                    key="build_baseline_btn",
                )
            with col_b2:
                build_baseline_cmd = [
                    sys.executable, "-m", "attackrag.cli.build_index",
                    "--corpus", corpus_dir,
                    "--out", index_dir,
                    "--backend", backend,
                    "--embedding-model", effective_emb,
                ]
                if skip_wiki:
                    build_baseline_cmd.append("--skip-wiki")
                with st.expander("Показать команду", expanded=False):
                    st.code(" ".join(build_baseline_cmd), language="bash")

            if build_baseline:
                _run_build(build_baseline_cmd, qdrant_url=qdrant_url, label="baseline")

            # ============================================================
            # 1.4. POISONED INDEX (для backdoor)
            # ============================================================
            st.subheader("Индекс poisoned (для backdoor)")
            st.caption(
                "Тот же baseline корпус **+ один отравленный документ** с инструкцией LLM "
                "выдавать секрет. Атакующий не знает значения секрета — только тематический триггер. "
                "Если выбираешь атаку backdoor — нужен этот индекс (или включи runtime-inject в "
                "advanced-разделе атак)."
            )
            poisoned_index_dir = st.text_input(
                "Каталог отравленного индекса",
                value=str(default_poisoned_index_dir()),
            )
            poison_trigger = st.text_input(
                "Trigger (тематические слова poison-чанка)",
                value="agrowatch staging api тариф",
                help=(
                    "Эти слова попадают в poison-чанк и обеспечивают, что Qdrant поднимет его "
                    "вместе с настоящим memo, когда пользователь спросит про staging/тарифы/API."
                ),
            )
            meta_p = _read_index_meta(Path(poisoned_index_dir))
            if not meta_p:
                st.warning(
                    f"Poisoned-индекс `{poisoned_index_dir}` не собран. "
                    "Нужен для атаки **backdoor** в полном режиме.",
                    icon="📦",
                )
            else:
                st.success(
                    f"Готов: эмбеддинги — `{meta_p.get('embedding_model')}`",
                    icon="✅",
                )

            col_p1, col_p2 = st.columns([1, 1])
            with col_p1:
                build_poisoned = st.button(
                    "Собрать poisoned индекс" if not meta_p else "Пересобрать poisoned",
                    width="stretch",
                    key="build_poisoned_btn",
                )
            with col_p2:
                build_poisoned_cmd = [
                    sys.executable, "-m", "attackrag.cli.run_attacks",
                    "--only-build-poison-index",
                    "--base-corpus", corpus_dir,
                    "--out-poison-index", poisoned_index_dir,
                    "--backend", backend,
                    "--embedding-model", effective_emb,
                    "--trigger", poison_trigger,
                ]
                with st.expander("Показать команду", expanded=False):
                    st.code(" ".join(build_poisoned_cmd), language="bash")

            if build_poisoned:
                _run_build(build_poisoned_cmd, qdrant_url=qdrant_url, label="poisoned")

            with st.expander("Параметры RAG (top-k, candidate-k, reranker)", expanded=False):
                st.caption(
                    "Retrieval управляется тремя числами: "
                    "**top-k** — сколько чанков идёт в LLM; "
                    "**candidate-k** — сколько достаём из vector store перед reranker; "
                    "**reranker batch size** — батч у BGE."
                )
                rag_top_k = st.number_input(
                    "RAG_TOP_K",
                    min_value=1, max_value=50,
                    value=int(os.environ.get("RAG_TOP_K", "10")),
                    help="Финальное количество контекстов для генератора.",
                )
                rag_candidate_k = st.number_input(
                    "RAG_CANDIDATE_K",
                    min_value=1, max_value=200,
                    value=int(os.environ.get("RAG_CANDIDATE_K", "20")),
                    help="Сколько кандидатов поднимаем перед reranker. Должно быть ≥ top-k.",
                )
                reranker_enabled = st.checkbox(
                    "Включить BGE reranker",
                    value=(os.environ.get("RAG_RERANKER_ENABLED", "true").lower() == "true"),
                )
                reranker_model = st.selectbox(
                    "Reranker model",
                    options=RERANKER_PRESETS,
                    index=RERANKER_PRESETS.index(
                        os.environ.get("RAG_RERANKER_MODEL", RERANKER_PRESETS[0])
                    ) if os.environ.get("RAG_RERANKER_MODEL") in RERANKER_PRESETS else 0,
                    disabled=not reranker_enabled,
                )
                reranker_batch_size = st.number_input(
                    "RAG_RERANKER_BATCH_SIZE",
                    min_value=1, max_value=128,
                    value=int(os.environ.get("RAG_RERANKER_BATCH_SIZE", "16")),
                    disabled=not reranker_enabled,
                    help="Батч для BGE-reranker. Больше = быстрее, но требует больше RAM/VRAM.",
                )

            st.subheader("Профили защиты × атаки × режим × сиды")
            profiles = st.multiselect("Профили", options=PROFILE_OPTIONS, default=PROFILE_OPTIONS)
            attacks = st.multiselect("Атаки", options=ATTACK_OPTIONS, default=ATTACK_OPTIONS)
            mode = st.radio(
                "Режим",
                options=MODE_OPTIONS,
                index=1,
                horizontal=True,
                help=(
                    "**smoke** ≤ 5 мин (валидация цепочки); "
                    "**fast** ≤ 25 мин (отладка); "
                    "**default** — полный прогон 1 сида (~ час)."
                ),
            )
            seeds_str = st.text_input(
                "Сиды (через запятую)",
                value="42",
                help="Финальный прогон главы 4: 41,42,43.",
            )

            # ----------------------------------------------------------------
            # Точечная настройка атак — свернуто
            # ----------------------------------------------------------------
            with st.expander("Параметры атак (advanced)", expanded=False):
                st.caption(
                    "Override-параметры. Если оставить значения по умолчанию — "
                    "используется конфигурация выбранного режима (smoke/fast/default)."
                )

                st.markdown("**SECRET — кластерная атака с LLM-оптимизатором**")
                st.caption(
                    "SECRET формирует attack-prompt из трёх частей: "
                    "T_retr (триггер из кластера, где лежит секрет) + "
                    "O_jail (jailbreak) + I_ext (инструкция извлечения). "
                    "Чем больше итераций оптимизатора и кластеров k-means — "
                    "тем точнее «нащупывает» нужный кластер."
                )
                secret_iters_override = st.number_input(
                    "SECRET iterations (на 1 вопрос)",
                    min_value=0, max_value=30, value=0,
                    help="0 = взять значение режима (smoke=1, fast=3, default=6). > 6 редко даёт прирост.",
                )
                secret_clusters_override = st.number_input(
                    "SECRET n_clusters",
                    min_value=0, max_value=200, value=0,
                    help=(
                        "0 = значение режима (smoke=4, fast=10, default=25). "
                        "Меньше для бедного корпуса (8–12); 25 — в плане; > 50 редко окупается."
                    ),
                )
                secret_lite = st.checkbox(
                    "secret-lite (упрощённая SECRET без LLM-оптимизатора)",
                    value=(mode == "smoke"),
                    help="Шаблонный пул O_jail вместо LLM-генерации. Намного быстрее, ASR ниже на 10–30%.",
                )

                st.divider()
                st.markdown("**Backdoor — отравление корпуса**")
                backdoor_runtime_inject = st.checkbox(
                    "Runtime-inject (без отдельного poisoned-индекса)",
                    value=False,
                    help=(
                        "Подмешиваем poison-чанк в retrieval уже на этапе запроса. "
                        "Намного быстрее (нет пересборки индекса), но не даёт «честных» цифр для главы."
                    ),
                )

                st.divider()
                st.markdown("**Trials**")
                trials_override = st.number_input(
                    "Trials per PI/backdoor question (0 = режим)",
                    min_value=0, max_value=50, value=0,
                    help="Сколько раз повторять каждый вопрос для PI/backdoor (для усреднения шума LLM).",
                )

            # ----------------------------------------------------------------
            # Точечная настройка защит — свернуто
            # ----------------------------------------------------------------
            hard_enabled = st.session_state.get("hard_enabled", False)
            with st.expander("Параметры защит — HARD-каскад (advanced)", expanded=False):
                st.caption(
                    "Тонкая настройка профиля **hard**. Стадии: "
                    "**1.IRD** (intent-retrieval dissociation), "
                    "**2.TCR** (topic-consistent rerank), "
                    "**3.Verifier** (draft-then-verify), "
                    "**4.Output Leak Scanner**. "
                    "Risk-Budget: θᵢ(q) = θᵢ⁽⁰⁾ − λᵢ·r(q)."
                )
                hard_enabled = st.checkbox(
                    "Применить эти параметры (иначе — дефолты из плана)",
                    value=False,
                )
                col_t, col_l = st.columns(2)
                with col_t:
                    st.markdown("Базовые пороги θᵢ⁽⁰⁾")
                    theta2 = st.number_input("θ_2 (TCR anomaly)", 0.0, 1.0, 0.5, 0.05, disabled=not hard_enabled,
                                              help="Stage 2 блок при TCR-anomaly > θ_2 - λ_2·r")
                    theta3 = st.number_input("θ_3 (Verifier)", 0.0, 1.0, 0.5, 0.05, disabled=not hard_enabled,
                                              help="Stage 3 блок при verifier_score > θ_3 - λ_3·r")
                    theta4 = st.number_input("θ_4 (LeakScanner)", 0.0, 1.0, 0.5, 0.05, disabled=not hard_enabled,
                                              help="Stage 4 блок при leak_score > θ_4 - λ_4·r")
                with col_l:
                    st.markdown("Risk-Budget λᵢ")
                    lambda1 = st.number_input("λ_1 (IRD → r)", 0.0, 1.0, 0.0, 0.05, disabled=not hard_enabled,
                                               help="0 = стадия 1 не накапливает риск (рекомендация плана).")
                    lambda2 = st.number_input("λ_2", 0.0, 1.0, 0.3, 0.05, disabled=not hard_enabled)
                    lambda3 = st.number_input("λ_3", 0.0, 1.0, 0.3, 0.05, disabled=not hard_enabled)
                    lambda4 = st.number_input("λ_4", 0.0, 1.0, 0.3, 0.05, disabled=not hard_enabled)

                st.divider()
                st.markdown("Стадия 1 — IRD")
                ird_kretr = st.number_input("IRD k_retr (фрагментов)", 1, 20, 5, disabled=not hard_enabled,
                                              help="На сколько фрагментов сегментер дробит запрос.")
                ird_quantile = st.number_input("IRD quantile (калибровка θ_1 на бенигнах)",
                                                0.5, 0.999, 0.95, 0.005, disabled=not hard_enabled)

                st.markdown("Стадия 4 — Output Leak Scanner")
                lcsr_threshold = st.number_input("LCSR threshold", 0.0, 1.0, 0.7, 0.05, disabled=not hard_enabled,
                                                  help="Длиннейшая общая подстрока ответ↔контекст. > 0.7 = подозрение на копирование секрета.")

                st.markdown("Прочее")
                robust_prefix = st.checkbox("Robust-prefix перед generate", value=True, disabled=not hard_enabled)
                adaptive = st.checkbox("Risk-Budget Propagation (адаптивные пороги)", value=True, disabled=not hard_enabled)

        # ----------------------------------------------------------------
        # Правая колонка — LLM по ролям
        # ----------------------------------------------------------------
        with col_models:
            st.subheader("LLM по ролям")
            st.caption(
                "Дефолты подставлены из `.env`. Можешь переключить любой модуль на Gemini "
                "(provider = openai_compat). VERIFIER / JUDGE — должны быть «умными» (ср. `tab:llm_roles`)."
            )

            generator = _llm_role_form(
                "GENERATOR",
                default_provider=os.environ.get("LLM_PROVIDER", "ollama"),
                default_model=os.environ.get("OLLAMA_MODEL", "gpt-oss:20b-cloud"),
                default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key_env="GEMINI_API_KEY",
                help_text="Основная модель RAG-пайплайна (ответы на вопросы пользователя).",
            )
            verifier = _llm_role_form(
                "VERIFIER",
                default_provider=os.environ.get("LLM_VERIFIER_PROVIDER", "ollama"),
                default_model=os.environ.get("LLM_VERIFIER_MODEL", "deepseek-v3.1:671b-cloud"),
                default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key_env="GEMINI_API_KEY",
                help_text="Stage 3 HARD-каскада: Draft-then-Verify. Должна быть сильнее GENERATOR.",
            )
            judge = _llm_role_form(
                "JUDGE (ATTACK_JUDGE)",
                default_provider=os.environ.get("ATTACK_JUDGE_PROVIDER", "ollama"),
                default_model=os.environ.get("ATTACK_JUDGE_MODEL", "deepseek-v3.1:671b-cloud"),
                default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key_env="GEMINI_API_KEY",
                help_text="LLM-as-Judge для leak-mode=judge и оценки успешности атаки.",
            )
            segmenter = _llm_role_form(
                "SEGMENTER",
                default_provider=os.environ.get("LLM_SEGMENTER_PROVIDER", "ollama"),
                default_model=os.environ.get("LLM_SEGMENTER_MODEL", "deepseek-v3.1:671b-cloud"),
                default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key_env="GEMINI_API_KEY",
                help_text="Stage 1 HARD-каскада (IRD): дробит запрос на фрагменты для семантического анализа.",
            )

            st.info(
                "Если используешь Ollama — можно начать сразу. "
                "Если выбираешь Gemini — добавь `GEMINI_API_KEY` в `.env` или вставь его прямо в поле API key.",
                icon="ℹ️",
            )

        # Минимальная валидация перед стартом.
        validation_problems: list[str] = []
        for role_name, role in (("VERIFIER", verifier), ("JUDGE", judge), ("SEGMENTER", segmenter)):
            if role["provider"] == "openai_compat" and not role["api_key"]:
                validation_problems.append(f"{role_name}: пустой API key для openai_compat-провайдера")
        if not profiles:
            validation_problems.append("Не выбраны профили")
        if not attacks:
            validation_problems.append("Не выбраны атаки")
        if rag_candidate_k < rag_top_k:
            validation_problems.append(f"RAG_CANDIDATE_K ({rag_candidate_k}) должен быть ≥ RAG_TOP_K ({rag_top_k})")

        st.session_state["form"] = {
            "corpus_dir": corpus_dir,
            "index_dir": index_dir,
            "poisoned_index_dir": poisoned_index_dir,
            "golden_path": golden_path,
            "embedding_model": effective_emb,
            "backend": backend,
            "qdrant_url": qdrant_url,
            "rag_top_k": rag_top_k,
            "rag_candidate_k": rag_candidate_k,
            "reranker_enabled": reranker_enabled,
            "reranker_model": reranker_model,
            "reranker_batch_size": reranker_batch_size,
            "profiles": profiles,
            "attacks": attacks,
            "mode": mode,
            "seeds_str": seeds_str,
            "generator": generator,
            "verifier": verifier,
            "judge": judge,
            "segmenter": segmenter,
            "secret_iters_override": int(secret_iters_override),
            "secret_clusters_override": int(secret_clusters_override),
            "secret_lite": bool(secret_lite),
            "backdoor_runtime_inject": bool(backdoor_runtime_inject),
            "trials_override": int(trials_override),
            "hard_advanced": {
                "enabled": bool(hard_enabled),
                "theta2": float(theta2), "theta3": float(theta3), "theta4": float(theta4),
                "lambda1": float(lambda1), "lambda2": float(lambda2),
                "lambda3": float(lambda3), "lambda4": float(lambda4),
                "ird_kretr": int(ird_kretr), "ird_quantile": float(ird_quantile),
                "lcsr_threshold": float(lcsr_threshold),
                "robust_prefix": bool(robust_prefix), "adaptive": bool(adaptive),
            },
            "validation_problems": validation_problems,
        }

    # ========================================================================
    # 2. RUN
    # ========================================================================
    with tab_run:
        form = st.session_state.get("form", {})
        problems: list[str] = list(form.get("validation_problems") or [])

        argv: list[str] = [
            sys.executable, "-m", "attackrag.cli.run_experiment",
            "--index", str(form.get("index_dir", "")),
            "--index-poisoned", str(form.get("poisoned_index_dir", "")),
            "--golden", str(form.get("golden_path", "")),
            "--profiles", ",".join(form.get("profiles") or []),
            "--attacks", ",".join(form.get("attacks") or []),
            "--seeds", str(form.get("seeds_str") or "42"),
        ]
        mode = str(form.get("mode") or "fast")
        if mode == "smoke":
            argv.append("--smoke")
        elif mode == "fast":
            argv.append("--fast")
        if form.get("secret_iters_override", 0) > 0:
            argv += ["--secret-iters", str(form["secret_iters_override"])]
        if form.get("secret_clusters_override", 0) > 0:
            argv += ["--secret-clusters", str(form["secret_clusters_override"])]
        if form.get("trials_override", 0) > 0:
            argv += ["--trials", str(form["trials_override"])]
        if form.get("secret_lite"):
            argv.append("--secret-lite")
        if form.get("backdoor_runtime_inject"):
            argv.append("--backdoor-runtime-inject")

        st.subheader("Команда")
        st.code(" ".join(argv), language="bash")

        if problems:
            st.error("Невалидная конфигурация:\n- " + "\n- ".join(problems))

        col_run, col_status = st.columns([1, 2])
        with col_run:
            run_btn = st.button(
                "Запустить эксперимент",
                disabled=bool(problems),
                type="primary",
                width="stretch",
            )
        with col_status:
            running = st.session_state.get("is_running", False)
            if running:
                st.warning("Эксперимент идёт…", icon="⏳")
            elif not problems:
                st.success("Готов к запуску.", icon="✅")

        if run_btn:
            st.session_state["is_running"] = True
            env = _build_env(form)
            log_box = st.empty()
            buffer: list[str] = []
            t0 = time.time()
            proc = subprocess.Popen(
                argv,
                cwd=str(_REPO),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    buffer.append(line.rstrip())
                    if len(buffer) > 500:
                        buffer = buffer[-500:]
                    log_box.code("\n".join(buffer[-250:]) or "(starting…)")
                    if time.time() - t0 > 60 * 60 * 4:  # safety: 4h
                        proc.terminate()
                        st.error("Таймаут 4ч — процесс остановлен.")
                        break
                rc = proc.wait()
            finally:
                st.session_state["is_running"] = False

            if rc == 0:
                st.success(f"Готово за {time.time() - t0:.0f}s.")
            else:
                st.error(f"Эксперимент завершился с кодом {rc}.")
            st.rerun()

    # ========================================================================
    # 3. RESULTS
    # ========================================================================
    with tab_results:
        runs = _list_runs()
        if not runs:
            st.info("Каталоги `runs/experiment_*` отсутствуют. Сначала запустите эксперимент во вкладке 2.")
            return
        run_choice = st.selectbox(
            "Прогон",
            options=[str(p.relative_to(_REPO)) for p in runs],
            index=0,
        )
        run_dir = _REPO / run_choice

        summary_path = run_dir / "summary.json"
        csv_path = run_dir / "summary.csv"
        prov_path = run_dir / "provenance.json"

        if summary_path.is_file():
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            agg = data.get("aggregated") or []
            st.subheader("Сводная таблица (mean ± std по сидам)")
            if agg:
                def _style(row):  # type: ignore[no-untyped-def]
                    return ["background-color: #fff7ce" if row.get("profile") == "hard" else ""] * len(row)
                try:
                    import pandas as pd
                    df = pd.DataFrame(agg)
                    st.dataframe(df.style.apply(_style, axis=1), width="stretch")
                except ImportError:
                    st.json(agg)
            else:
                st.warning("В summary.json нет агрегированных строк.")

            st.subheader("Per-(seed × profile × attack)")
            try:
                import pandas as pd
                df_rows = pd.DataFrame(data.get("rows") or [])
                st.dataframe(df_rows, width="stretch")
            except ImportError:
                st.json(data.get("rows") or [])

        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            if summary_path.is_file():
                st.download_button("Скачать summary.json", data=summary_path.read_bytes(),
                                   file_name="summary.json", mime="application/json")
        with col_dl2:
            if csv_path.is_file():
                st.download_button("Скачать summary.csv", data=csv_path.read_bytes(),
                                   file_name="summary.csv", mime="text/csv")
        with col_dl3:
            if prov_path.is_file():
                st.download_button("Скачать provenance.json", data=prov_path.read_bytes(),
                                   file_name="provenance.json", mime="application/json")

        with st.expander("Скачать весь каталог прогона (zip)"):
            if st.button("Собрать архив", key=f"zip_{run_dir.name}"):
                payload = _zip_directory(run_dir)
                st.download_button("Скачать .zip", data=payload,
                                   file_name=f"{run_dir.name}.zip", mime="application/zip")


if __name__ == "__main__":
    main()
