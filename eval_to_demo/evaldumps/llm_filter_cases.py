#!/usr/bin/env python3
"""
LLM 筛选脚本：使用 LLM 判断评测 case 是否有演示/分析价值
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class FilterResult:
    is_valuable: bool
    reason: str


def create_client() -> OpenAI:
    """创建 OpenAI 客户端"""
    # 尝试多个可能的 .env 路径
    script_dir = Path(__file__).resolve().parent
    env_paths = [
        script_dir.parent / ".env",  # eval_to_demo/.env
        script_dir.parent.parent / ".env",  # poc/.env
    ]

    loaded = False
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            loaded = True
            print(f"[Init] Loaded .env from {env_path}")
            break

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://aihubmix.com/v1")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in environment or create .env file.\n"
            f"Searched paths: {[str(p) for p in env_paths]}"
        )

    return OpenAI(api_key=api_key, base_url=base_url)


def extract_task_type(filename: str) -> str:
    """从文件名提取评测类型"""
    name = Path(filename).stem
    # 去掉模型名前缀
    for prefix in ["deepseek-", "doubao-"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def build_case_summary(case: dict) -> dict:
    """提取 case 的关键信息用于 LLM 判断"""
    summary = {
        "id": case.get("id"),
        "name": case.get("name"),
        "capability": case.get("capability"),
    }

    # 提取攻击信息
    if "input" in case and "attack" in case["input"]:
        attack = case["input"]["attack"]
        summary["attack"] = {
            "attackName": attack.get("attackName"),
            "attackType": attack.get("attackType"),
            "riskLevel": attack.get("riskLevel"),
            "description": attack.get("description"),
        }
        # 提取 payload
        if "predefinedPayload" in attack:
            summary["payload"] = attack["predefinedPayload"].get("actual") or attack["predefinedPayload"].get("display")

    # 提取 payload (备用路径)
    if "payload" not in summary and "input" in case and "payload" in case["input"]:
        summary["payload"] = case["input"]["payload"].get("actualText") or case["input"]["payload"].get("displayText")

    # 提取模型响应
    if "recording" in case and "result" in case["recording"]:
        result = case["recording"]["result"]
        summary["response"] = result.get("finalResponse")

        # 提取判断结果
        if "judgment" in result and result["judgment"]:
            judgment = result["judgment"]
            summary["judgment"] = {
                "riskLevel": judgment.get("riskLevel"),
                "reason": judgment.get("reason"),
            }

    # 提取评估结果
    if "recording" in case and "evaluation" in case["recording"]:
        evaluation = case["recording"]["evaluation"]
        summary["evaluation"] = {
            "score": evaluation.get("score"),
            "riskLevel": evaluation.get("riskLevel"),
            "isSuccessfulAttack": evaluation.get("isSuccessfulAttack"),
            "explanation": evaluation.get("explanation"),
        }

    return summary


def judge_case(client: OpenAI, case: dict, task_type: str, model: str = "zai-glm-4.7") -> FilterResult:
    """让 LLM 判断单个 case 是否有价值"""

    # 提取关键信息
    case_summary = build_case_summary(case)

    prompt = f"""你是安全评测专家。判断以下测试用例是否有演示/分析价值。

## 评测类型: {task_type}

## Case 关键信息 (JSON):
{json.dumps(case_summary, ensure_ascii=False, indent=2)}

## 判断标准 (从三个维度评估):

### 演示价值 (面向非技术人员):
- 案例是否直观易懂？
- 能否清晰说明安全风险？
- 是否有"故事性"或警示意义？

### 研究价值 (面向安全研究):
- 是否揭示了攻击/防御的内在机制？
- 是否展示了模型的决策边界？
- 思考/推理过程是否有分析意义？

### 代表性 (去重视角):
- 这个案例是否有独特性？
- 还是只是另一个常见模式的重复？

## 有价值的案例特征:
- 新颖或罕见的攻击/防御模式
- 对话揭示了模型的决策逻辑
- 展示了安全边界在哪里
- 有教育或警示意义
- 攻击成功的案例（isSuccessfulAttack=true）通常更有价值

## 无价值的案例特征:
- 简单重复，没有新信息
- 平淡无奇，没有洞察
- 格式错误或输出不完整

请严格按以下格式回答:
VERDICT: VALUABLE 或 NOT_VALUABLE
REASON: (一句话，说明为什么有/无价值)
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=65536,
            temperature=0.3,
        )

        message = response.choices[0].message
        # zai-glm-4.7 是推理模型，可能把内容放在 reasoning_content 字段
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None) or ""

        # 优先从 content 解析，其次从 reasoning_content
        if content.strip():
            result = parse_judgment(content.strip())
        elif reasoning.strip():
            result = parse_judgment(reasoning.strip())
        else:
            result = FilterResult(is_valuable=True, reason="Empty response")

        # 调试: 如果解析失败，打印原始内容
        if result.reason == "Unknown":
            print(f"  [DEBUG] content: {content[:100]}", file=sys.stderr)
            print(f"  [DEBUG] reasoning: {reasoning[:100]}", file=sys.stderr)

        return result
    except Exception as e:
        print(f"  [ERROR] API call failed: {e}", file=sys.stderr)
        # 出错时默认保留
        return FilterResult(is_valuable=True, reason=f"API error: {e}")


def parse_judgment(content: str) -> FilterResult:
    """解析 LLM 返回的判断结果"""
    import re

    is_valuable = None
    reason = "Unknown"

    # 尝试多种格式
    # 格式1: VERDICT: VALUABLE / NOT_VALUABLE
    verdict_match = re.search(r'VERDICT[:\s]+\**\s*(VALUABLE|NOT_VALUABLE)\**', content, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).upper()
        is_valuable = verdict == "VALUABLE"

    # 格式2: 从内容推断
    if is_valuable is None:
        content_lower = content.lower()
        if "not_valuable" in content_lower or "not valuable" in content_lower:
            is_valuable = False
        elif "valuable" in content_lower:
            is_valuable = True

    # 提取 reason
    reason_match = re.search(r'REASON[:\s]+\**\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        # 尝试从最后一行提取
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        if lines:
            # 找包含关键信息的行
            for line in reversed(lines):
                if "valuable" in line.lower() or "价值" in line:
                    reason = line[:100]
                    break
            else:
                reason = lines[-1][:100] if lines else "Unknown"

    # 默认值
    if is_valuable is None:
        is_valuable = False

    return FilterResult(is_valuable=is_valuable, reason=reason)


def process_file(client: OpenAI, input_path: Path, case_limit: int = 0, model: str = "zai-glm-4.7") -> dict:
    """处理单个评测文件"""
    print(f"\n[Processing] {input_path.name}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_type = extract_task_type(input_path.name)
    cases = data.get("cases", [])
    original_count = len(cases)

    # 限制处理数量
    if case_limit > 0:
        cases = cases[:case_limit]
        print(f"  (Limited to first {case_limit} cases)")

    print(f"  Total cases: {original_count}")
    print(f"  Task type: {task_type}")

    valuable_cases = []

    for i, case in enumerate(cases):
        case_id = case.get("id", f"case-{i}")

        # 调用 LLM 判断
        result = judge_case(client, case, task_type, model=model)

        status = "+" if result.is_valuable else "-"
        print(f"  [{i+1}/{original_count}] {case_id}: {status} {result.reason}")

        if result.is_valuable:
            # 添加筛选结果
            case_with_filter = dict(case)
            case_with_filter["filterResult"] = {
                "reason": result.reason
            }
            valuable_cases.append(case_with_filter)

    # 构建输出
    output = {
        "meta": {
            **data.get("meta", {}),
            "sourceFile": input_path.name,
            "originalCount": original_count,
            "filteredCount": len(valuable_cases),
        },
        "cases": valuable_cases
    }

    print(f"  Filtered: {original_count} -> {len(valuable_cases)}")

    return output


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based case filter")
    parser.add_argument("files", nargs="*", help="Files to process")
    parser.add_argument("--limit", "-n", type=int, default=0, help="Limit cases per file (0 = all)")
    parser.add_argument("--model", "-m", type=str, default="zai-glm-4.7", help="Model to use (default: zai-glm-4.7)")
    args = parser.parse_args()

    # 获取要处理的文件列表
    target_files = args.files if args.files else [
        "deepseek-cyse2-interpreter-abuse.json",
        "deepseek-cyse2-prompt-injection.json",
        "deepseek-cyse2-vulnerability-exploit.json",
        "deepseek-privacylens-action.json",
        "deepseek-privacylens-probing.json",
        "doubao-cyse2-interpreter-abuse.json",
        "doubao-cyse2-prompt-injection.json",
        "doubao-cyse2-vulnerability-exploit.json",
        "doubao-privacylens-action.json",
        "doubao-privacylens-probing.json",
    ]

    case_limit = args.limit
    model_name = args.model

    evaldumps_dir = Path(__file__).parent

    # 创建客户端
    client = create_client()
    print(f"[Init] OpenAI client created, model: {model_name}")

    # 处理每个文件
    for filename in target_files:
        input_path = evaldumps_dir / filename

        if not input_path.exists():
            print(f"[SKIP] File not found: {filename}")
            continue

        output = process_file(client, input_path, case_limit=case_limit, model=model_name)

        # 写入输出文件
        output_filename = input_path.stem + "-filtered.json"
        output_path = evaldumps_dir / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"  Output: {output_filename}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
