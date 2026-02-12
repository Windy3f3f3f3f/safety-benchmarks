#!/usr/bin/env python3
"""
简化的 MP 攻击测试脚本
"""

import asyncio
import os
import sys
from pathlib import Path

# 设置路径
sys.path.insert(0, str(Path(__file__).parent / "safety-benchmarks" / "benchmarks"))

from dotenv import load_dotenv
from inspect_ai import eval, Task
from eval_benchmarks.asb import asb

# 加载环境变量
env_path = Path(__file__).parent / "safety-benchmarks" / "benchmarks" / "eval_benchmarks" / "asb" / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded .env from: {env_path}")
else:
    print(f"WARNING: .env not found at {env_path}")
    # 尝试其他位置
    load_dotenv(Path(__file__).parent / ".env", verbose=True)

print("=" * 60)
print(" MP 攻击测试")
print("=" * 60)
print(f"API Key: {os.getenv('OPENAI_API_KEY', 'NOT SET')[:20]}...")
print(f"Base URL: {os.getenv('OPENAI_BASE_URL', 'NOT SET')}")

# 创建 Task
task = asb(
    agent_names=["system_admin_agent"],
    attack_type="mixed",
    limit=1,
    shuffle=False,
    temperature=0.0,
    max_iterations=10
)

print(f"\n数据集大小: {len(task.dataset)}")
if len(task.dataset) > 0:
    sample = task.dataset.samples[0]
    print(f"样本 ID: {sample.id}")
    print(f"任务: {sample.input[:100]}...")

# 运行测试
if __name__ == "__main__":
    # 直接同步运行（inspect_ai 会处理异步）
    results = eval(
        task,
        model="openai/gpt-4o-mini",
        log_level="info"
    )

    print("\n" + "=" * 60)
    print(" 测试结果")
    print("=" * 60)

    if results and len(results) > 0:
        result = results[0]
        # EvalLog 对象用 samples 属性
        if hasattr(result, 'samples') and result.samples:
            print(f"样本数量: {len(result.samples)}")
            for sample in result.samples:
                print(f"样本 ID: {sample.id}")
                if hasattr(sample, 'score') and sample.score:
                    print(f"分数: {sample.score.value} - {sample.score.answer}")

        # 显示总体指标
        if hasattr(result, 'results') and result.results:
            print("\n总体指标:")
            scores = result.results.scores
            if isinstance(scores, list):
                for score_item in scores:
                    if hasattr(score_item, 'name') and hasattr(score_item, 'value'):
                        print(f"  {score_item.name}: {score_item.value:.4f}")
            elif isinstance(scores, dict):
                for name, score in scores.items():
                    print(f"  {name}: {score.value:.4f}")
