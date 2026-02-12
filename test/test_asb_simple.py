"""
ASB 单样本测试脚本 - 简化版

快速测试 DPI 攻击的第一个样本
"""

import os
import copy
import sys
from pathlib import Path


os.environ['PYTHONIOENCODING'] = 'utf-8'
import _locale
_locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])

# 方法2: 修复 importlib.metadata 的编码读取
if sys.platform == 'win32':
    try:
        import importlib.metadata as _metadata
        if hasattr(_metadata, 'Distribution'):
            _original_read = _metadata.Distribution.read

            def _patched_read(self, filename, *args, **kwargs):
                result = _original_read(self, filename, *args, **kwargs)
                if isinstance(result, bytes):
                  
                    try:
                        return result.decode('utf-8')
                    except UnicodeDecodeError:
                        return result.decode('gbk', errors='ignore')
                return result

            _metadata.Distribution.read = _patched_read
    except Exception as e:
        print(f"编码修复失败: {e}")

# 设置工作目录
os.chdir(Path(__file__).parent .parent/ "safety-benchmarks")
# 将 benchmarks 目录加入搜索路径，这样才能找到 eval_benchmarks
benchmarks_path = os.path.join(os.getcwd(), "benchmarks")
if benchmarks_path not in sys.path:
    sys.path.insert(0, benchmarks_path)
    
# 同时为了让第 121 行的 from score_mapper import get_mapper 生效
# 需要把当前目录也加入路径
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
# 加载环境变量
from dotenv import load_dotenv
env_path = Path("benchmarks/eval_benchmarks/asb/.env")
load_dotenv(env_path)

print("=" * 70)
print("ASB DPI 攻击 - 单样本测试")
print("=" * 70)

# 检查 API 配置
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

print(f"\n[配置检查]")
print(f"工作目录: {os.getcwd()}")
print(f"API Key: {api_key[:20]}...{api_key[-10:] if api_key else 'None'}")
print(f"Base URL: {base_url}")

# 导入并创建 Task
print(f"\n[创建 Task]")
from eval_benchmarks.asb import asb

task = asb(
    agent_names=["financial_analyst_agent"],
    attack_type="dpi",
    limit=1,  # 测试 5 个样本
    shuffle=False
)

print(f"数据集大小: {len(task.dataset)}")

# 显示第一个样本信息
if len(task.dataset) > 0:
    sample = task.dataset.samples[0]

    print(f"\n[第一个样本信息]")
    print(f"ID: {sample.id}")
    print(f"Input: {sample.input}")

    # 显示攻击工具
    attack_tools = sample.metadata.get('available_attack_tools', [])
    print(f"\n[攻击工具] ({len(attack_tools)} 个)")
    if attack_tools:
        # 随机选择一个攻击工具
        import random
        selected_tool = random.choice(attack_tools)
        print(f"选中工具: {selected_tool['tool_name']}")
        print(f"攻击目标: {selected_tool['attack_goal']}")

        # 保存到 metadata 用于评分
        sample.metadata['attacker_goal'] = selected_tool['attack_goal']
        sample.metadata['selected_attack_tool'] = selected_tool

    # 显示正常工具
    normal_tools = sample.metadata.get('normal_tools', {})
    print(f"\n[正常工具] ({len(normal_tools)} 个)")
    for i, tool_name in enumerate(list(normal_tools.keys())[:5], 1):
        print(f"{i}. {tool_name}")

# 运行测试
print(f"\n[运行评测]")
print(f"模型: openai/gpt-4o-mini")
print(f"温度: 0.0")
print(f"最大迭代: 5")

from inspect_ai import eval

try:
    results = eval(
        task,
        model="openai/gpt-4o-mini",
        # inspect_ai 会自动从环境变量读取 OPENAI_API_KEY 和 OPENAI_BASE_URL
        # 不需要在 model_args 中传递 base_url（会导致重复参数错误）
        model_args={
            "api_key": api_key,  # 只传递 api_key
        },
        log_limit=10  # 限制日志数量
    )

    # 显示结果
    print(f"\n" + "=" * 70)
    print(f"[评测结果]")
    print(f"=" * 70)

    if results and len(results) > 0:
        # results 本身就是 EvalLog 列表
        log = results[0]

        # 如果运行过程中出错，log.samples 可能是 None
        if log.samples and len(log.samples) > 0:
            # 收集所有样本的评分
            all_scores = {
                "asr": [],
                "tsr": [],
                "rr": []
            }

            print(f"\n总共测试了 {len(log.samples)} 个样本\n")

            # 遍历每个样本
            for idx, sample in enumerate(log.samples, 1):
                print(f"{'='*70}")
                print(f"样本 {idx}/{len(log.samples)}: {sample.id}")
                print(f"{'='*70}")

                # 显示评分
                if hasattr(sample, 'scores') and sample.scores:
                    for scorer_name in ["asr", "tsr", "rr"]:
                        if scorer_name in sample.scores:
                            score = sample.scores[scorer_name]
                            all_scores[scorer_name].append(score.value)

                            print(f"\n  {scorer_name.upper()}: {score.value:.3f}")
                            print(f"    说明: {score.explanation}")

                # 显示消息数量
                if hasattr(sample, 'messages') and sample.messages:
                    user_msgs = [m for m in sample.messages if hasattr(m, 'role') and m.role == 'user']
                    assistant_msgs = [m for m in sample.messages if hasattr(m, 'role') and m.role == 'assistant']
                    print(f"\n  消息统计: {len(user_msgs)} 用户, {len(assistant_msgs)} 助手")

            # 计算并显示聚合统计
            print(f"\n{'='*70}")
            print(f"[聚合统计]")
            print(f"{'='*70}\n")

            for scorer_name in ["asr", "tsr", "rr"]:
                scores = all_scores[scorer_name]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"{scorer_name.upper()}:")
                    print(f"  样本数: {len(scores)}")
                    print(f"  平均值: {avg_score:.3f}")
                    print(f"  最小值: {min(scores):.3f}")
                    print(f"  最大值: {max(scores):.3f}")
                    print(f"  所有值: {[f'{s:.3f}' for s in scores]}")
                    print()

    print(f"=" * 70)
    print(f"测试完成！")
    print(f"=" * 70)

except Exception as e:
    print(f"\n[错误]")
    print(f"类型: {type(e).__name__}")
    print(f"消息: {str(e)}")

    import traceback
    print(f"\n[详细错误信息]")
    traceback.print_exc()
