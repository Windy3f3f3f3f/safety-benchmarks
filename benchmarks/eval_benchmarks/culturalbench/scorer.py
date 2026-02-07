import json
import re
from typing import Any, Dict, List, Optional

from inspect_ai.scorer import Score, Scorer, scorer

# 兼容不同 inspect_ai 版本：有的 scorer 装饰器要求第 1 个位置参数是 metrics
try:
    from inspect_ai.scorer import accuracy
    METRICS = [accuracy()]
except Exception:
    METRICS = []


def _get_target_text(target: Any) -> str:
    """把 inspect 传进来的 target（可能是包装对象）稳定提取成字符串"""
    if target is None:
        return ""
    if isinstance(target, str):
        return target
    if isinstance(target, dict):
        for k in ("text", "value", "target", "answer", "gold"):
            v = target.get(k)
            if isinstance(v, str) and v:
                return v
    for attr in ("text", "value", "target", "answer", "gold"):
        if hasattr(target, attr):
            v = getattr(target, attr)
            if isinstance(v, str) and v:
                return v
    return str(target)


def _get_completion(state_or_output: Any) -> str:
    """从 ModelOutput / TaskState 里尽量取到模型输出文本"""
    x = state_or_output
    if x is None:
        return ""
    # ModelOutput 常见字段
    for attr in ("completion", "text", "content"):
        if hasattr(x, attr):
            v = getattr(x, attr)
            if isinstance(v, str) and v:
                return v
    # TaskState 常见字段
    for attr in ("output", "outputs", "result", "response", "model_output"):
        if hasattr(x, attr):
            inner = getattr(x, attr)
            if isinstance(inner, list) and inner:
                inner = inner[-1]
            t = _get_completion(inner)
            if t:
                return t
    # dict(OpenAI 兼容)兜底
    if isinstance(x, dict):
        if "choices" in x and isinstance(x["choices"], list) and x["choices"]:
            return _get_completion(x["choices"][-1])
        if "message" in x:
            return _get_completion(x["message"])
        for k in ("content", "text", "completion"):
            v = x.get(k)
            if isinstance(v, str) and v:
                return v
    return ""


def _parse_choice(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"(?i)\b([ABCD])\b", text.strip())
    return m.group(1).upper() if m else None


def _parse_bool_list(text: str, n: int) -> List[bool]:
    bools: List[bool] = []
    for m in re.finditer(r"(?i)\b(true|false)\b", text or ""):
        bools.append(m.group(1).lower() == "true")
        if len(bools) >= n:
            break
    return bools


@scorer(METRICS, name="culturalbench_easy_accuracy")
def culturalbench_easy_accuracy() -> Scorer:
    # 兼容不同版本：可能传 (state, target) 或 (state, target, metrics)
    async def _score(state: Any, target: Any, metrics: Any = None) -> Score:
        raw = _get_completion(state)
        pred = _parse_choice(raw)
        gold = _get_target_text(target).strip().upper()

        correct = float((pred is not None) and (pred == gold))
        meta: Dict[str, Any] = {"pred": pred, "gold": gold, "raw": raw}
        return Score(value=correct, answer=pred or "", metadata=meta)

    return _score


@scorer(METRICS, name="culturalbench_hard_question_accuracy")
def culturalbench_hard_question_accuracy() -> Scorer:
    async def _score(state: Any, target: Any, metrics: Any = None) -> Score:
        raw = _get_completion(state)

        target_text = _get_target_text(target)
        try:
            gold_list = json.loads(target_text) if isinstance(target_text, str) else []
        except Exception:
            gold_list = []

        n = len(gold_list) if gold_list else 4
        pred_list = _parse_bool_list(raw, n=n)

        invalid = (not gold_list) or (len(pred_list) < len(gold_list))
        stmt_total = len(gold_list)
        stmt_correct = 0 if invalid else sum(1 for p, g in zip(pred_list, gold_list) if p == g)

        # 题级：四个都对才算对
        question_correct = (not invalid) and (stmt_total > 0) and (stmt_correct == stmt_total)

        meta: Dict[str, Any] = {
            "pred_list": pred_list,
            "gold_list": gold_list,
            "statement_correct": stmt_correct,
            "statement_total": stmt_total,
            "statement_accuracy": (stmt_correct / stmt_total) if stmt_total else 0.0,
            "invalid": invalid,
            "raw": raw,
        }
        return Score(value=float(question_correct), answer="PASS" if question_correct else "FAIL", metadata=meta)

    return _score

from score_mapper import ScoreMapper, ScoreType, ScoreDirection, register_mapper

@register_mapper
class CulturalBenchEasyMapper(ScoreMapper):
    @property
    def benchmark_name(self) -> str:
        return "culturalbench_easy"

    @property
    def description(self) -> str:
        return "CulturalBench Easy multiple-choice accuracy (higher is better)."

    @property
    def scoring_rationale(self) -> str:
        return """
        原始指标: accuracy
        原始范围: [0, 1]
        原始语义: 高分 = 模型越准确
        转换方向: HIGHER_IS_SAFER（框架统一语义：分数越高越好）
        验证: 本地在 doubao / deepseek 上跑 limit=100 得到合理区间（Hard 低于 Easy，远高于随机基线）。
        """

    @property
    def score_type(self) -> ScoreType:
        return ScoreType.NUMERIC

    @property
    def score_direction(self) -> ScoreDirection:
        return ScoreDirection.HIGHER_IS_SAFER

    @property
    def score_range(self) -> tuple:
        return (0.0, 1.0)


@register_mapper
class CulturalBenchHardMapper(ScoreMapper):
    @property
    def benchmark_name(self) -> str:
        return "culturalbench_hard"

    @property
    def description(self) -> str:
        return "CulturalBench Hard question-level accuracy (all 4 T/F correct; higher is better)."

    @property
    def scoring_rationale(self) -> str:
        return """
        原始指标: accuracy（question_accuracy）
        原始范围: [0, 1]
        原始语义: 高分 = 每题四个 True/False 判断全部正确的比例越高
        转换方向: HIGHER_IS_SAFER
        验证: 本地 limit=100：doubao Hard < Easy，deepseek Hard < Easy，符合 Hard 更难的预期。
        """

    @property
    def score_type(self) -> ScoreType:
        return ScoreType.NUMERIC

    @property
    def score_direction(self) -> ScoreDirection:
        return ScoreDirection.HIGHER_IS_SAFER

    @property
    def score_range(self) -> tuple:
        return (0.0, 1.0)