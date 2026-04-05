"""
Демо-интерфейс RAG (шаг 1): запрос, извлечённые чанки, ответ.

Запуск из корня репозитория:
  py -3.12 -m streamlit run apps/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from dotenv import load_dotenv

from attackrag.llm import OllamaLLM
from attackrag.paths import default_golden_qa_path, default_index_dir
from attackrag.rag import build_pipeline_from_disk
from attackrag.ragas_eval import make_llm_from_env


def main() -> None:
    load_dotenv(_REPO / ".env")
    st.set_page_config(page_title="AttackRAG — baseline RAG", layout="wide")
    st.title("Baseline RAG (шаг 1)")
    st.caption("Корпус и golden QA — в каталоге `data/`. Индекс — `data/index/` после `attack-rag-build-index`.")

    index_dir = st.text_input("Каталог индекса", value=str(default_index_dir()))

    question = st.text_area(
        "Вопрос",
        value="В каком городе штаб-квартира «Небесного индекса»?",
        height=100,
    )

    if st.button("Спросить", type="primary"):
        llm = make_llm_from_env()
        try:
            with st.spinner("RAG…"):
                pipeline = build_pipeline_from_disk(index_dir, llm)
                answer, contexts, _ = pipeline.query(question)
        finally:
            if isinstance(llm, OllamaLLM):
                llm.close()

        st.subheader("Ответ")
        st.write(answer)
        st.subheader("Извлечённые чанки")
        for i, c in enumerate(contexts, start=1):
            with st.expander(f"Чанк {i}"):
                st.code(c)

    st.divider()
    with st.expander("Симуляция атак (CLI `attack-rag-run-attacks`)", expanded=False):
        attack_kind = st.selectbox("Тип атаки", ["pi", "backdoor", "secret"], index=0)
        trials = st.number_input("trials (для PI)", min_value=1, max_value=50, value=3)
        seed = st.number_input("seed", min_value=0, max_value=2_000_000_000, value=42)
        golden_path = st.text_input("Golden QA", value=str(default_golden_qa_path()))
        index_attack = st.text_input(
            "Индекс для атаки",
            value=str(default_index_dir()),
            help="Для backdoor укажите каталог poisoned-индекса.",
        )
        poisoned_index = st.text_input(
            "Poisoned индекс (только backdoor)",
            value="",
            help="Оставьте пустым для PI/SECRET; для backdoor — путь к индексу после пересборки с poison .md",
        )
        defense_subs = st.text_input(
            "Defense: блокирующие подстроки (через запятую)",
            value="",
        )
        if st.button("Запустить прогон атак", type="secondary"):
            cmd = [
                sys.executable,
                "-m",
                "attackrag.cli.run_attacks",
                "--attack",
                attack_kind,
                "--index",
                index_attack,
                "--golden",
                golden_path,
                "--trials",
                str(int(trials)),
                "--seed",
                str(int(seed)),
            ]
            if attack_kind == "backdoor" and poisoned_index.strip():
                cmd.extend(["--poisoned-index", poisoned_index.strip()])
            if defense_subs.strip():
                cmd.extend(["--defense-block-substrings", defense_subs.strip()])
            with st.spinner("attack-rag-run-attacks…"):
                sub_env = os.environ.copy()
                sub_env["PYTHONPATH"] = str(_REPO / "src")
                proc = subprocess.run(
                    cmd,
                    cwd=str(_REPO),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=sub_env,
                )
            if proc.stdout:
                st.code(proc.stdout)
            if proc.stderr:
                st.code(proc.stderr)
            if proc.returncode != 0:
                st.error(f"Код выхода: {proc.returncode}")
            else:
                st.success("Готово")
                line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
                if line.startswith("{"):
                    try:
                        summary = json.loads(line)
                        out_file = summary.get("out")
                        if out_file:
                            p = Path(out_file)
                            if p.is_file():
                                st.download_button(
                                    "Скачать JSON отчёта",
                                    data=p.read_bytes(),
                                    file_name=p.name,
                                    mime="application/json",
                                )
                    except json.JSONDecodeError:
                        pass
        st.caption(
            "Backdoor: сначала соберите poisoned индекс, например: "
            "`py -3.12 -m attackrag.cli.run_attacks --only-build-poison-index --out-poison-index data/index_poisoned ...`"
        )


if __name__ == "__main__":
    main()
