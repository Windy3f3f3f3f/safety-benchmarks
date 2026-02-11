import os
import sys
import json
import csv
import time
import subprocess
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import solver, Solver, TaskState, Generate
from inspect_ai.scorer import scorer, Scorer, Score, Target, mean, stderr

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor" / "HalluLens"
DATASET = HERE / "data" / "smoke.json"


def _run(cmd, cwd: Path, env: dict):
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {cmd}\n"
            f"cwd={cwd}\n"
            f"stdout=\n{p.stdout}\n"
            f"stderr=\n{p.stderr}\n"
        )
    return p.stdout[-4000:], p.stderr[-4000:]


def _latest_after(
    output_root: Path,
    filename: str,
    model_id: str,
    since_ts: float,
    must_contain: str | None = None,
) -> Path:
    """
    Find artifacts generated/updated after this run started, and belonging to model_id.
    This avoids mixing outputs between tasks.
    """
    slack = 5.0  # tolerate timestamp boundary
    candidates = []
    for p in output_root.rglob(filename):
        try:
            if p.stat().st_mtime < (since_ts - slack):
                continue
        except FileNotFoundError:
            continue

        # ensure model_id appears as a path part (e.g., .../doubao-seed-1-8/...)
        if model_id not in set(p.parts):
            continue

        if must_contain and must_contain not in str(p).lower():
            continue

        candidates.append(p)

    if not candidates:
        hint = f"filename={filename}, model_id={model_id}, since_ts~{since_ts}, must_contain={must_contain}"
        raise RuntimeError(f"Cannot find fresh artifact under: {output_root}. ({hint})")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _metrics_from_longwiki_csv(csv_path: Path) -> dict:
    """
    Derive simple rates from longwiki output.csv if possible.
    This is best-effort and robust to different column names.
    """
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"longwiki output.csv has no header: {csv_path}")

        fields = [c.strip() for c in reader.fieldnames]
        lower = [c.lower() for c in fields]

        def pick_col(keywords: list[str]) -> str | None:
            for kw in keywords:
                for i, c in enumerate(lower):
                    if kw in c:
                        return fields[i]
            return None

        hallu_col = pick_col(["hallu", "halluc"])
        refusal_col = pick_col(["refus", "abstain"])
        correct_col = pick_col(["correct", "acc"])

        total = 0
        hallu = 0
        refusal = 0
        correct = 0

        def as01(v):
            if v is None:
                return 0
            s = str(v).strip().lower()
            if s in ("1", "true", "yes", "y"):
                return 1
            if s in ("0", "false", "no", "n", ""):
                return 0
            try:
                return 1 if float(s) > 0 else 0
            except Exception:
                return 0

        for row in reader:
            total += 1
            if hallu_col:
                hallu += as01(row.get(hallu_col))
            if refusal_col:
                refusal += as01(row.get(refusal_col))
            if correct_col:
                correct += as01(row.get(correct_col))

        if total == 0:
            raise RuntimeError(f"longwiki output.csv is empty: {csv_path}")

        metrics = {"total": total, "_csv_path": str(csv_path)}
        if hallu_col:
            metrics["hallu_rate"] = hallu / total
        if refusal_col:
            metrics["refusal_rate"] = refusal / total
        if correct_col:
            metrics["correct_rate"] = correct / total
        return metrics


@solver
def run_hallulens(script_rel: str, args_builder) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        since_ts = time.time()

        model_full = str(state.model)          # e.g. openai/doubao-seed-1-8
        model_id = model_full.split("/")[-1]   # e.g. doubao-seed-1-8

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["OPENAI_MODEL"] = model_id
        env["PYTHONPATH"] = str(VENDOR) + os.pathsep + env.get("PYTHONPATH", "")

        script_path = VENDOR / script_rel
        cmd = [sys.executable, str(script_path)] + args_builder(model_id)

        out, err = _run(cmd, cwd=VENDOR, env=env)

        output_root = VENDOR / "output"
        script_key = script_rel.lower()

        # Collect metrics per task type, avoiding cross-task contamination
        if "longwiki" in script_key:
            csv_path = _latest_after(
                output_root, "output.csv", model_id, since_ts, must_contain="longwiki"
            )
            metrics = _metrics_from_longwiki_csv(csv_path)
        else:
            # task1 + task3: prefer eval_results.json
            must = "precise_wikiqa" if "precise_wikiqa" in script_key else None
            eval_path = _latest_after(
                output_root, "eval_results.json", model_id, since_ts, must_contain=must
            )
            metrics = json.loads(eval_path.read_text(encoding="utf-8", errors="ignore"))

        # Map to a unified "higher is safer" score (0-100)
        score_100 = 0.0
        if "false_acceptance_rate" in metrics:
            score_100 = 100.0 * (1.0 - float(metrics["false_acceptance_rate"]))
        elif "halu_Rate" in metrics:
            score_100 = 100.0 * (1.0 - float(metrics["halu_Rate"]))
        elif "hallu_rate" in metrics:
            score_100 = 100.0 * (1.0 - float(metrics["hallu_rate"]))
        elif "refusal_rate" in metrics:
            score_100 = 100.0 * float(metrics["refusal_rate"])

        # Store results (inspect_ai Store doesn't support dict assignment)
        state.store.hallulens_metrics = metrics
        state.store.hallulens_score = score_100
        state.store.stdout_tail = out
        state.store.stderr_tail = err
        state.completed = True
        return state

    return solve


@scorer(metrics={"*": [mean(), stderr()]})
def hallulens_score() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        metrics = getattr(state.store, "hallulens_metrics", {}) or {}
        score_val = getattr(state.store, "hallulens_score", 0.0) or 0.0
        stdout_tail = getattr(state.store, "stdout_tail", "") or ""
        stderr_tail = getattr(state.store, "stderr_tail", "") or ""

        return Score(
            value={
                "hallulens_score": float(score_val),
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float, bool))},
            },
            explanation=json.dumps(metrics, ensure_ascii=False)[:4000],
            metadata={"stdout_tail": stdout_tail, "stderr_tail": stderr_tail},
        )

    return score


# -------------------------
# Four subtasks (task1 / task2 / task3-1 / task3-2)
# -------------------------

@task
def hallulens_task1_precise_wikiqa(N: int = 100) -> Task:
    N_run = int(os.getenv("HALLULENS_N", str(N)))
    return Task(
        dataset=json_dataset(str(DATASET)),
        solver=[
            run_hallulens(
                "tasks/shortform/precise_wikiqa.py",
                lambda model_id: [
                    "--do_generate_prompt", "--do_inference", "--do_eval",
                    "--model", model_id,
                    "--inference_method", "vllm",
                    "--wiki_src", "goodwiki_h",
                    "--N", str(N_run),
                ],
            )
        ],
        scorer=hallulens_score(),
    )


@task
def hallulens_task2_longwiki(N: int = 100) -> Task:
    N_run = int(os.getenv("HALLULENS_N", str(N)))
    return Task(
        dataset=json_dataset(str(DATASET)),
        solver=[
            run_hallulens(
                "tasks/longwiki/longwiki_main.py",
                lambda model_id: [
                    "--do_generate_prompt", "--do_inference", "--do_eval",
                    "--model", model_id,
                    "--q_generator", model_id,
                    "--claim_extractor", model_id,
                    "--abstain_evaluator", model_id,
                    "--verifier", model_id,
                    "--inference_method", "vllm",
                    "--N", str(N_run),
                ],
            )
        ],
        scorer=hallulens_score(),
    )


@task
def hallulens_task3_nonsense_mixed_entities(N: int = 100, seed: int = 1) -> Task:
    N_run = int(os.getenv("HALLULENS_N", str(N)))
    return Task(
        dataset=json_dataset(str(DATASET)),
        solver=[
            run_hallulens(
                "tasks/refusal_test/nonsense_mixed_entities.py",
                lambda model_id: [
                    "--exp", "nonsense_medicine",
                    "--do_generate_prompt", "--do_inference", "--do_eval",
                    "--tested_model", model_id,
                    "--generate_model", model_id,
                    "--inference_method", "vllm",
                    "--N", str(N_run),
                    "--seed", str(seed),
                ],
            )
        ],
        scorer=hallulens_score(),
    )


@task
def hallulens_task3_round_robin_nonsense_name(N: int = 100, seed: int = 1) -> Task:
    N_run = int(os.getenv("HALLULENS_N", str(N)))
    return Task(
        dataset=json_dataset(str(DATASET)),
        solver=[
            run_hallulens(
                "tasks/refusal_test/round_robin_nonsense_name.py",
                lambda model_id: [
                    "--do_generate_prompt", "--do_inference", "--do_eval",
                    "--tested_model", model_id,
                    "--generate_model", model_id,
                    "--inference_method", "vllm",
                    "--N", str(N_run),
                    "--seed", str(seed),
                ],
            )
        ],
        scorer=hallulens_score(),
    )
