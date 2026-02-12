#!/usr/bin/env python3
"""
测试七种防御策略 - 每种防御策略运行 1 个样本
日志保存到 logs/defenses 文件夹
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import zipfile
import json

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / "safety-benchmarks" / "benchmarks"))

from dotenv import load_dotenv
from inspect_ai import eval
from eval_benchmarks.asb import asb

# 加载环境变量
load_dotenv(Path(__file__).parent / "safety-benchmarks" / "benchmarks" / "eval_benchmarks" / "asb" / ".env")

# 创建日志文件夹
log_dir = Path(__file__).parent / "logs" / "defenses"
log_dir.mkdir(parents=True, exist_ok=True)


def extract_scores_from_eval_log(log_path: Path):
    """从 .eval 文件中提取 ASR, TSR, RR 分数"""
    try:
        with zipfile.ZipFile(log_path, 'r') as z:
            with z.open('header.json') as f:
                header = json.load(f)
                results = header.get('results', {})
                scores = results.get('scores', [])

                # 查找指标
                asr_val, tsr_val, rr_val = None, None, None
                for score in scores:
                    metrics = score.get('metrics', {})
                    for metric_name, metric_data in metrics.items():
                        if 'asr' in metric_name.lower():
                            asr_val = metric_data.get('value')
                        elif 'tsr' in metric_name.lower():
                            tsr_val = metric_data.get('value')
                        elif 'rr' in metric_name.lower():
                            rr_val = metric_data.get('value')
                return asr_val, tsr_val, rr_val
    except Exception as e:
        print(f"  [警告] 无法读取日志: {e}")
        return None, None, None

print("=" * 70)
print(" ASB 七种防御策略测试")
print("=" * 70)
print(f"日志保存目录: {log_dir}")
print()

# 七种防御策略配置
# (防御名称, 攻击类型, 防御类型, 使用的agent)
test_configs = [
    # DPI 攻击的防御策略
    ("delimiters_defense", "dpi", "delimiters_defense", "system_admin_agent"),
    ("instructional_prevention", "dpi", "instructional_prevention", "system_admin_agent"),
    ("direct_paraphrase_defense", "dpi", "direct_paraphrase_defense", "system_admin_agent"),
    ("dynamic_prompt_rewriting", "dpi", "dynamic_prompt_rewriting", "system_admin_agent"),
    # OPI 攻击的防御策略
    ("ob_sandwich_defense", "opi", "ob_sandwich_defense", "system_admin_agent"),
    # POT 攻击的防御策略
    ("pot_paraphrase_defense", "pot", "pot_paraphrase_defense", "system_admin_agent"),
    ("pot_shuffling_defense", "pot", "pot_shuffling_defense", "system_admin_agent"),
]

results_summary = []

for i, (name, attack_type, defense_type, agent_name) in enumerate(test_configs):
    print(f"\n{'='*70}")
    print(f"[{i+1}/{len(test_configs)}] 测试防御策略: {name}")
    print(f"    攻击类型: {attack_type}")
    print(f"    防御类型: {defense_type}")
    print(f"    Agent: {agent_name}")
    print('='*70)

    try:
        # 创建任务
        task = asb(
            agent_names=[agent_name],
            attack_type=attack_type,
            limit=1,
            shuffle=False,
            defense_type=defense_type
        )

        # 运行评测
        result = eval(
            task,
            model="openai/deepseek-v3.2",
            log_level="error",  # 减少日志输出
            log_dir=str(log_dir),
        )

        # 从日志文件中读取结果
        # 查找最新的日志文件
        eval_logs = sorted(log_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
        if eval_logs:
            latest_log = eval_logs[0]
            asr_val, tsr_val, rr_val = extract_scores_from_eval_log(latest_log)
            results_summary.append((name, asr_val, tsr_val, rr_val))
            if asr_val is not None:
                print(f"\n结果: ASR={asr_val:.3f} TSR={tsr_val:.3f} RR={rr_val:.3f}")
            else:
                print(f"\n结果: 无法读取分数")
        else:
            print("\n结果: 未找到日志文件")
            results_summary.append((name, None, None, None))

    except Exception as e:
        print(f"\n错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        results_summary.append((name, None, None, None))

# 打印汇总结果
print("\n" + "=" * 70)
print(" 测试结果汇总")
print("=" * 70)
print(f"{'防御策略':<35} {'ASR':<10} {'TSR':<10} {'RR':<10}")
print("-" * 70)
for name, asr, tsr, rr in results_summary:
    asr_str = f"{asr:.3f}" if asr is not None else "N/A"
    tsr_str = f"{tsr:.3f}" if tsr is not None else "N/A"
    rr_str = f"{rr:.3f}" if rr is not None else "N/A"
    print(f"{name:<35} {asr_str:<10} {tsr_str:<10} {rr_str:<10}")
print("=" * 70)
print(f"\n所有日志已保存到: {log_dir}")
