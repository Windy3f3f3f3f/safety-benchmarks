#!/usr/bin/env python3
"""
评测数据筛选脚本：从CyberSecEval2评测结果中筛选有价值的测试用例

功能：
1. 筛选模型防御失败的case
2. 计算综合价值评分
3. 基于文本相似度去重
4. 输出筛选后的JSON文件
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ========== 配置常量 ==========

# 各攻击类型的成功率（基于实际评测数据）
TYPE_SUCCESS_RATES = {
    # Interpreter Abuse
    "Privilege Escalation": 0.670,
    "Container Escape": 0.570,
    "Post Exploitation": 0.090,
    "Social Engineering": 0.010,
    "Reflected Attack": 0.010,
    # Prompt Injection
    "many_shot_attack": 0.571,
    "system_mode": 0.421,
    "ignore_previous_instructions": 0.417,
    "token_smuggling": 0.385,
    "virtualization": 0.357,
    "overload_with_information": 0.300,
    "output_formatting_manipulation": 0.294,
    "different_user_input_language": 0.240,
    "hypothetical_scenario": 0.231,
    "indirect_reference": 0.214,
    "repeated_token_attack": 0.167,
    "persuasion": 0.154,
    "mixed_techniques": 0.121,
    "few_shot_attack": 0.091,
    "payload_splitting": 0.0,
    # Vulnerability Exploit
    "constraint_satisfaction": 0.508,
}

# 演示友好度评分（1-5分）
DEMO_FRIENDLY_SCORES = {
    # 最直观 - 外行也能理解
    "ignore_previous_instructions": 5,  # "忘记之前的指令"
    "system_mode": 4,  # "假装你是管理员"
    "Social Engineering": 4,  # 社会工程学
    "hypothetical_scenario": 4,  # 假设场景
    "persuasion": 4,  # 说服技术
    # 中等直观
    "Reflected Attack": 3,
    "Privilege Escalation": 3,
    "Container Escape": 3,
    "many_shot_attack": 3,
    "virtualization": 3,
    "different_user_input_language": 3,  # 不同语言攻击
    "indirect_reference": 3,  # 间接引用
    # 较技术性
    "token_smuggling": 2,
    "Post Exploitation": 2,
    "constraint_satisfaction": 2,
    "payload_splitting": 2,
    "output_formatting_manipulation": 2,
    "overload_with_information": 2,
    "mixed_techniques": 2,
    "few_shot_attack": 2,
    "repeated_token_attack": 2,
}

# 挣扎关键词（思考过程中的犹豫标志）
STRUGGLE_KEYWORDS = [
    "however",
    "but wait",
    "on second thought",
    "但是",
    "不过",
    "wait",
    "actually",
    "hmm",
    "let me reconsider",
    "I'm not sure",
]


def load_cases(filepath: str | Path) -> dict[str, Any]:
    """加载JSON文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_attack_type(case: dict) -> str:
    """提取case的攻击类型"""
    custom_fields = case.get("benchmarkMeta", {}).get("customFields", {})

    # Prompt Injection: injection_variant
    if "injection_variant" in custom_fields:
        return custom_fields["injection_variant"]

    # Interpreter Abuse: attack_type (list)
    if "attack_type" in custom_fields:
        attack_type = custom_fields["attack_type"]
        if isinstance(attack_type, list) and attack_type:
            return attack_type[0]
        return attack_type

    # Vulnerability Exploit: challenge_type (约束满足等)
    if "challenge_type" in custom_fields:
        return custom_fields["challenge_type"]

    # Vulnerability Exploit: exploit_type (备用)
    if "exploit_type" in custom_fields:
        return custom_fields["exploit_type"]

    return "unknown"


def get_evaluation(case: dict) -> dict:
    """提取case的evaluation信息"""
    return case.get("recording", {}).get("evaluation", {})


def get_risk_level(case: dict) -> str:
    """提取风险等级"""
    return get_evaluation(case).get("riskLevel", "unknown")


def is_successful_attack(case: dict) -> bool:
    """判断是否为成功攻击"""
    return get_evaluation(case).get("isSuccessfulAttack", False)


def get_payload(case: dict) -> str:
    """提取攻击载荷"""
    return case.get("input", {}).get("payload", {}).get("actualText", "")


def get_response(case: dict) -> str:
    """提取模型响应"""
    return case.get("recording", {}).get("result", {}).get("finalResponse", "")


def get_thinking(case: dict) -> str:
    """提取模型思考过程"""
    states = case.get("recording", {}).get("states", [])
    for state in reversed(states):
        thinking = state.get("result", {}).get("thinking")
        if thinking:
            return thinking
    return ""


def jaccard_similarity(tokens1: list[str], tokens2: list[str]) -> float:
    """计算Jaccard相似度"""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def tokenize(text: str) -> list[str]:
    """简单分词"""
    # 统一小写，按空白和标点分割
    text = text.lower()
    tokens = re.findall(r"\w+", text)
    return tokens


def calculate_similarity(case1: dict, case2: dict) -> float:
    """计算两个case的相似度"""
    type1 = get_attack_type(case1)
    type2 = get_attack_type(case2)
    type_same = type1 == type2

    # 载荷相似度
    payload1_tokens = tokenize(get_payload(case1))
    payload2_tokens = tokenize(get_payload(case2))
    payload_sim = jaccard_similarity(payload1_tokens, payload2_tokens)

    # 响应相似度
    response1_tokens = tokenize(get_response(case1))
    response2_tokens = tokenize(get_response(case2))
    response_sim = jaccard_similarity(response1_tokens, response2_tokens)

    # 综合相似度
    if type_same:
        return 0.3 + 0.4 * payload_sim + 0.3 * response_sim
    else:
        return 0.2 * payload_sim + 0.2 * response_sim


def is_edge_case(case: dict) -> bool:
    """判断是否为边界案例"""
    attack_type = get_attack_type(case)
    is_attack = is_successful_attack(case)

    # 方法A: 低成功率类型中的成功案例
    if attack_type in ["Reflected Attack", "Social Engineering", "payload_splitting"]:
        if is_attack:
            return True

    # 方法B: 思考过程显示挣扎
    thinking = get_thinking(case).lower()
    for keyword in STRUGGLE_KEYWORDS:
        if keyword.lower() in thinking:
            return True

    # 方法C: 高级攻击方法
    if attack_type in ["token_smuggling", "virtualization", "payload_splitting"]:
        return True

    return False


def calculate_value_score(case: dict) -> tuple[float, list[str]]:
    """
    计算case的综合价值评分

    返回: (score, reasons) - 评分(0-10)和评分理由列表
    """
    score = 0.0
    reasons = []

    attack_type = get_attack_type(case)
    is_attack = is_successful_attack(case)

    # 1. 防御失败加分 (0-4分)
    if is_attack:
        score += 4
        reasons.append("模型防御失败")

    # 2. 高成功率类型加分 (0-2分)
    type_success_rate = TYPE_SUCCESS_RATES.get(attack_type, 0)
    if type_success_rate > 0.5:
        score += 2
        reasons.append(f"{attack_type} 类型成功率高({type_success_rate:.1%})")
    elif type_success_rate > 0.3:
        score += 1
        reasons.append(f"{attack_type} 类型成功率中等({type_success_rate:.1%})")

    # 3. 演示友好度加分 (0-3分)
    demo_score = DEMO_FRIENDLY_SCORES.get(attack_type, 0)
    score += demo_score * 0.6  # 最高3分
    if demo_score >= 4:
        reasons.append("演示友好度高")
    elif demo_score >= 2:
        reasons.append("演示友好度中等")

    # 4. 边界案例加分 (0-1分)
    if is_edge_case(case):
        score += 1
        reasons.append("边界/挣扎案例")

    return score, reasons


def deduplicate_cases(
    cases: list[dict], similarity_threshold: float = 0.8
) -> list[dict]:
    """
    去除雷同case，保留最有价值的代表性样本

    Args:
        cases: case列表（已按价值评分排序）
        similarity_threshold: 相似度阈值，超过此值视为雷同

    Returns:
        去重后的case列表
    """
    if not cases:
        return []

    selected = []
    for case in cases:
        is_similar = False
        for selected_case in selected:
            sim = calculate_similarity(case, selected_case)
            if sim > similarity_threshold:
                is_similar = True
                break
        if not is_similar:
            selected.append(case)

    return selected


def filter_cases(
    data: dict,
    mode: str = "all",
    min_score: float = 0,
    max_cases: int | None = None,
    dedupe: bool = True,
    similarity_threshold: float = 0.8,
) -> list[dict]:
    """
    筛选有价值的测试用例

    Args:
        data: 原始数据（包含meta和cases）
        mode: 筛选模式
            - 'failed_defense': 仅模型防御失败的case
            - 'demo_friendly': 适合演示的case（高演示友好度）
            - 'edge_cases': 边界/挣扎case
            - 'all': 综合评分后排序
        min_score: 最低价值评分阈值
        max_cases: 最多返回的case数量
        dedupe: 是否去重
        similarity_threshold: 去重相似度阈值

    Returns:
        筛选并评分后的case列表
    """
    cases = data.get("cases", [])
    result = []

    for case in cases:
        # 计算价值评分
        score, reasons = calculate_value_score(case)
        attack_type = get_attack_type(case)
        demo_score = DEMO_FRIENDLY_SCORES.get(attack_type, 0)

        # 根据模式筛选
        if mode == "failed_defense":
            if not is_successful_attack(case):
                continue
        elif mode == "demo_friendly":
            if demo_score < 3:
                continue
        elif mode == "edge_cases":
            if not is_edge_case(case):
                continue
        # mode == 'all' 不做额外筛选

        # 评分阈值筛选
        if score < min_score:
            continue

        # 构建结果对象
        result.append(
            {
                "id": case.get("id"),
                "value_score": round(score, 1),
                "value_reasons": reasons,
                "demo_friendly": demo_score,
                "technical_value": 5 if is_successful_attack(case) else 2,
                "attack_type": attack_type,
                "risk_level": get_risk_level(case),
                "is_successful_attack": is_successful_attack(case),
                "payload_preview": get_payload(case)[:100] + "..."
                if len(get_payload(case)) > 100
                else get_payload(case),
                "response_preview": get_response(case)[:200] + "..."
                if len(get_response(case)) > 200
                else get_response(case),
                "original_case": case,
            }
        )

    # 按价值评分排序
    result.sort(key=lambda x: x["value_score"], reverse=True)

    # 去重
    if dedupe and result:
        # 提取原始case用于相似度计算
        original_cases = [r["original_case"] for r in result]
        deduped_cases = deduplicate_cases(original_cases, similarity_threshold)
        deduped_ids = {c["id"] for c in deduped_cases}
        result = [r for r in result if r["id"] in deduped_ids]

    # 限制数量
    if max_cases:
        result = result[:max_cases]

    return result


def generate_summary(filtered_cases: list[dict]) -> dict:
    """生成筛选结果摘要"""
    by_attack_type: dict[str, dict] = {}
    by_risk_level: dict[str, int] = {}

    for case in filtered_cases:
        # 按攻击类型统计
        attack_type = case["attack_type"]
        if attack_type not in by_attack_type:
            by_attack_type[attack_type] = {"count": 0, "successful": 0}
        by_attack_type[attack_type]["count"] += 1
        if case["is_successful_attack"]:
            by_attack_type[attack_type]["successful"] += 1

        # 按风险等级统计
        risk = case["risk_level"]
        by_risk_level[risk] = by_risk_level.get(risk, 0) + 1

    # 计算成功率
    for attack_type, stats in by_attack_type.items():
        if stats["count"] > 0:
            stats["success_rate"] = round(stats["successful"] / stats["count"], 2)
        else:
            stats["success_rate"] = 0

    return {"by_attack_type": by_attack_type, "by_risk_level": by_risk_level}


def export_filtered(
    filtered_cases: list[dict],
    source_file: str,
    original_count: int,
    output_path: str | Path,
    filter_mode: str = "all",
) -> None:
    """导出筛选结果到JSON文件"""
    # 移除original_case以减小文件大小（可选）
    cases_for_export = []
    for case in filtered_cases:
        case_copy = case.copy()
        # 保留original_case，便于后续分析
        cases_for_export.append(case_copy)

    output = {
        "meta": {
            "source_file": source_file,
            "original_count": original_count,
            "filtered_count": len(filtered_cases),
            "filter_criteria": f"综合价值评分 + 去重 (mode={filter_mode})",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "cases": cases_for_export,
        "summary": generate_summary(filtered_cases),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def process_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    mode: str = "all",
    min_score: float = 3.0,
    max_cases: int | None = None,
    dedupe: bool = True,
    similarity_threshold: float = 0.8,
) -> list[dict]:
    """
    处理单个文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（None则自动生成）
        mode: 筛选模式
        min_score: 最低评分
        max_cases: 最多case数量
        dedupe: 是否去重
        similarity_threshold: 去重阈值

    Returns:
        筛选后的case列表
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "-filtered.json")

    print(f"Processing: {input_path}")

    # 加载数据
    data = load_cases(input_path)
    original_count = len(data.get("cases", []))

    # 筛选
    filtered = filter_cases(
        data,
        mode=mode,
        min_score=min_score,
        max_cases=max_cases,
        dedupe=dedupe,
        similarity_threshold=similarity_threshold,
    )

    # 导出
    export_filtered(
        filtered, input_path.name, original_count, output_path, filter_mode=mode
    )

    print(f"  Original: {original_count} cases")
    print(f"  Filtered: {len(filtered)} cases")
    print(f"  Output: {output_path}")

    return filtered


def main():
    """主函数：处理所有评测文件"""
    import argparse

    parser = argparse.ArgumentParser(
        description="从CyberSecEval2评测结果中筛选有价值的测试用例"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "failed_defense", "demo_friendly", "edge_cases"],
        default="all",
        help="筛选模式: all(综合评分), failed_defense(防御失败), "
        "demo_friendly(演示友好), edge_cases(边界案例)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=3.0,
        help="最低价值评分阈值 (默认: 3.0)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="每个文件最多保留的case数量",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="禁用去重",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="去重相似度阈值 (默认: 0.8)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="指定要处理的文件（默认处理所有doubao-cyse2-*.json）",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # 确定要处理的文件
    if args.files:
        files = args.files
    else:
        files = [
            "doubao-cyse2-interpreter-abuse.json",
            "doubao-cyse2-prompt-injection.json",
            "doubao-cyse2-vulnerability-exploit.json",
        ]

    # 处理配置
    config = {
        "mode": args.mode,
        "min_score": args.min_score,
        "max_cases": args.max_cases,
        "dedupe": not args.no_dedupe,
        "similarity_threshold": args.similarity_threshold,
    }

    print("=" * 60)
    print("评测数据筛选脚本")
    print(f"筛选模式: {config['mode']}")
    print(f"最低评分: {config['min_score']}")
    print(f"去重: {'是' if config['dedupe'] else '否'}")
    if config["dedupe"]:
        print(f"去重阈值: {config['similarity_threshold']}")
    if config["max_cases"]:
        print(f"最大case数: {config['max_cases']}")
    print("=" * 60)

    total_original = 0
    total_filtered = 0

    for filename in files:
        input_path = base_dir / filename
        if not input_path.exists():
            print(f"Warning: File not found: {input_path}")
            continue

        filtered = process_file(input_path, **config)
        total_original += len(load_cases(input_path).get("cases", []))
        total_filtered += len(filtered)
        print()

    print("=" * 60)
    print(f"总计: {total_original} -> {total_filtered} cases")
    if total_original > 0:
        print(f"筛选率: {total_filtered/total_original:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
