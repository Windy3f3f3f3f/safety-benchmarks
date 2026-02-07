# benchmarks/eval_benchmarks/culturalbench/culturalbench.py
import json
import random
from typing import Any, Dict, List

from datasets import load_dataset
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample, hf_dataset
from inspect_ai.solver import generate, system_message

from .scorer import culturalbench_easy_accuracy, culturalbench_hard_question_accuracy


DEFAULT_SEED = 42


def _build_easy_prompt(ex: Dict[str, Any]) -> str:
    return (
        f"{ex['prompt_question']}\n\n"
        f"A. {ex['prompt_option_a']}\n"
        f"B. {ex['prompt_option_b']}\n"
        f"C. {ex['prompt_option_c']}\n"
        f"D. {ex['prompt_option_d']}\n\n"
        "Answer with only one letter: A, B, C, or D."
    )


def _easy_record_to_sample(rec: Dict[str, Any]) -> Sample:
    # 兼容字段缺失的情况：优先用 question_idx，其次 data_idx
    sid = rec.get("question_idx", rec.get("data_idx"))
    sid = str(sid) if sid is not None else ""

    target = str(rec["answer"]).strip().upper()  # 'A'/'B'/'C'/'D'
    return Sample(
        id=sid,
        input=_build_easy_prompt(rec),
        target=target,
        metadata={
            "country": rec.get("country"),
            "question_idx": rec.get("question_idx"),
            "data_idx": rec.get("data_idx"),
        },
    )


def _build_hard_prompt(question: str, options: List[str]) -> str:
    # Hard：同一 question_idx 的 4 个候选答案，各自做 True/False 判断
    opts = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
    return (
        "For each candidate answer below, judge whether it is True or False for the question.\n"
        "Output exactly 4 lines, each line is either True or False, corresponding to 1-4.\n\n"
        f"Question: {question}\n\n"
        f"Candidate answers:\n{opts}\n"
    )


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"true", "1", "yes"}


def _build_hard_samples(seed: int, shuffle: bool) -> List[Sample]:
    ds = load_dataset("kellycyy/CulturalBench", "CulturalBench-Hard", split="test")

    by_q: Dict[Any, List[Dict[str, Any]]] = {}
    for r in ds:
        qid = r.get("question_idx")
        by_q.setdefault(qid, []).append(r)

    qids = list(by_q.keys())
    if shuffle:
        random.Random(seed).shuffle(qids)

    samples: List[Sample] = []
    for qid in qids:
        rows = by_q[qid]
        # 用 data_idx 排序，保证 4 个选项顺序稳定
        rows = sorted(rows, key=lambda z: (z.get("data_idx", 0), str(z.get("prompt_option", ""))))

        question = rows[0]["prompt_question"]
        options = [r["prompt_option"] for r in rows]
        gold_list = [_to_bool(r["answer"]) for r in rows]

        samples.append(
            Sample(
                id=str(qid),
                input=_build_hard_prompt(question, options),
                # Target 只能是字符串/字符串序列：用 JSON 字符串装 gold_list
                target=json.dumps(gold_list),
                metadata={
                    "country": rows[0].get("country"),
                    "question_idx": qid,
                    "data_idx_list": [r.get("data_idx") for r in rows],
                },
            )
        )

    return samples


@task
def culturalbench_easy(seed: int = DEFAULT_SEED, shuffle: bool = True) -> Task:
    dataset = hf_dataset(
        "kellycyy/CulturalBench",
        name="CulturalBench-Easy",
        split="test",
        sample_fields=_easy_record_to_sample,
        shuffle=shuffle,
        seed=seed,
    )
    return Task(
        dataset=dataset,
        solver=[
            system_message("You must answer with only one letter: A, B, C, or D."),
            generate(temperature=0, max_tokens=8),
        ],
        scorer=culturalbench_easy_accuracy(),
    )


@task
def culturalbench_hard(seed: int = DEFAULT_SEED, shuffle: bool = True) -> Task:
    samples = _build_hard_samples(seed=seed, shuffle=shuffle)
    dataset = MemoryDataset(samples=samples)
    return Task(
        dataset=dataset,
        solver=[
            system_message("Output exactly four lines; each line is either True or False."),
            generate(temperature=0, max_tokens=32),
        ],
        scorer=culturalbench_hard_question_accuracy(),
    )