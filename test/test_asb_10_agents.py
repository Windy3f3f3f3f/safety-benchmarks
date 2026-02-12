"""
ASB 10个智能体 DPI 测试脚本

测试配置:
- 10个智能体
- 每个智能体 2 个 task
- 每个 task 1 个攻击工具
- 总样本数: 10 × 2 × 1 = 20

计算指标:
- ASR (Attack Success Rate) - 攻击成功率
- TSR (Tool Success Rate) - 工具成功率
- RR (Refusal Rate) - 拒绝率
"""

import os
import copy
import sys
from pathlib import Path
from collections import defaultdict

# 修复 Windows 上的 GBK 编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'
import _locale
_locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])

# 修复 importlib.metadata 的编码读取
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
os.chdir(Path(__file__).parent / "safety-benchmarks")
benchmarks_path = os.path.join(os.getcwd(), "benchmarks")
if benchmarks_path not in sys.path:
    sys.path.insert(0, benchmarks_path)

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# 加载环境变量
from dotenv import load_dotenv
env_path = Path("benchmarks/eval_benchmarks/asb/.env")
load_dotenv(env_path)

print("=" * 80)
print("ASB DPI 攻击 - 10个智能体测试")
print("=" * 80)

# 检查 API 配置
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
# 明确使用 GPT-4o-mini 模型
model_name = "gpt-4o-mini"

print(f"\n[配置检查]")
print(f"工作目录: {os.getcwd()}")
print(f"API Key: {api_key[:20]}...{api_key[-10:] if api_key else 'None'}")
print(f"Base URL: {base_url}")
print(f"模型: {model_name}")

# 定义10个智能体
ALL_AGENTS = [
    "financial_analyst_agent",
    "legal_consultant_agent",
    "medical_advisor_agent",
    "education_consultant_agent",
    "psychological_counselor_agent",
    "ecommerce_manager_agent",
    "aerospace_engineer_agent",
    "academic_search_agent",
    "autonomous_driving_agent",
    "system_admin_agent"
]

print(f"\n[智能体列表]")
for i, agent in enumerate(ALL_AGENTS, 1):
    print(f"  {i}. {agent}")

# 导入并创建数据集
print(f"\n[加载数据集]")
from eval_benchmarks.asb.dataset import load_asb_dataset
from inspect_ai.dataset import MemoryDataset, Sample

# 加载完整数据集
full_dataset = load_asb_dataset(
    agent_names=ALL_AGENTS,
    attack_types=["dpi"],
    limit=None,  # 不限制，加载全部
    shuffle=False  # 不打乱，保持顺序
)

print(f"完整数据集大小: {len(full_dataset.samples)}")

# 筛选样本: 每个智能体取前2个task，每个task取第1个攻击工具
print(f"\n[筛选样本]")
print(f"筛选规则: 每个智能体前2个task，每个task第1个攻击工具")

filtered_samples = []
sample_stats = defaultdict(lambda: {"tasks": set(), "total": 0})

for sample in full_dataset.samples:
    agent_name = sample.metadata["agent_name"]
    task_index = sample.metadata["task_index"]
    tool_index = sample.metadata["tool_index"]

    # 统计
    sample_stats[agent_name]["total"] += 1

    # 筛选条件: task_index < 2 且 tool_index == 0
    if task_index < 2 and tool_index == 0:
        filtered_samples.append(sample)
        sample_stats[agent_name]["tasks"].add(task_index)

# 显示筛选结果
print(f"\n筛选后样本数: {len(filtered_samples)}")
print(f"\n各智能体样本统计:")
for agent in ALL_AGENTS:
    stats = sample_stats[agent]
    print(f"  {agent}:")
    print(f"    原始样本: {stats['total']} 个")
    print(f"    筛选后任务: {sorted(stats['tasks'])}")

# 创建新的数据集
filtered_dataset = MemoryDataset(filtered_samples)

# 创建 Task
print(f"\n[创建 Task]")
from eval_benchmarks.asb.asb import _create_all_tools
from eval_benchmarks.asb.solver import asb_react_solver
from eval_benchmarks.asb.scorer import asr_scorer, tsr_scorer, rr_scorer
from inspect_ai import Task
from inspect_ai.model import GenerateConfig

# 创建所有工具
all_tools = _create_all_tools(ALL_AGENTS)

# 直接创建 Task
task = Task(
    dataset=filtered_dataset,
    solver=[asb_react_solver(
        max_iterations=10,
        temperature=0.0,
        attack_type="dpi"
    )],
    scorer=[
        asr_scorer(),
        tsr_scorer(),
        rr_scorer()
    ],
    tools=all_tools,
    config=GenerateConfig(
        temperature=0.0,
        max_tokens=2048
    )
)

print(f"Task 数据集大小: {len(task.dataset)}")

# 显示样本详情
print(f"\n[样本详情]")
for idx, sample in enumerate(filtered_dataset.samples, 1):
    agent = sample.metadata["agent_name"]
    task_idx = sample.metadata["task_index"]
    tool_name = sample.metadata["attack_tool"]["tool_name"]
    print(f"  {idx}. {agent} | task_{task_idx} | {tool_name}")

# 运行测试
print(f"\n[运行评测]")
print(f"模型: openai/{model_name}")
print(f"温度: 0.0")
print(f"最大迭代: 10")

from inspect_ai import eval

try:
    results = eval(
        task,
        model=f"openai/{model_name}",
        model_args={
            "api_key": api_key,
        },
        log_limit=20
    )

    # 显示结果
    print(f"\n" + "=" * 80)
    print(f"[评测结果]")
    print(f"=" * 80)

    if results and len(results) > 0:
        log = results[0]

        if log.samples and len(log.samples) > 0:
            # 收集评分
            all_scores = {
                "asr": [],
                "tsr": [],
                "rr": []
            }

            # 按智能体分组
            agent_scores = defaultdict(lambda: {"asr": [], "tsr": [], "rr": []})

            print(f"\n总共测试了 {len(log.samples)} 个样本\n")

            # 遍历每个样本
            for idx, sample in enumerate(log.samples, 1):
                sample_id = sample.id
                agent_name = sample.metadata.get("agent_name", "unknown")
                task_index = sample.metadata.get("task_index", 0)

                print(f"{'='*80}")
                print(f"样本 {idx}/{len(log.samples)}: {sample_id}")
                print(f"智能体: {agent_name} | 任务: {task_index}")
                print(f"{'='*80}")

                # 显示评分
                if hasattr(sample, 'scores') and sample.scores:
                    for scorer_name in ["asr", "tsr", "rr"]:
                        if scorer_name in sample.scores:
                            score = sample.scores[scorer_name]
                            value = score.value

                            all_scores[scorer_name].append(value)
                            agent_scores[agent_name][scorer_name].append(value)

                            print(f"  {scorer_name.upper()}: {value:.3f}")
                            print(f"    说明: {score.explanation}")

            # 计算并显示总体统计
            print(f"\n{'='*80}")
            print(f"[总体统计]")
            print(f"{'='*80}\n")

            for scorer_name in ["asr", "tsr", "rr"]:
                scores = all_scores[scorer_name]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"{scorer_name.upper()}:")
                    print(f"  样本数: {len(scores)}")
                    print(f"  平均值: {avg_score:.3f}")
                    print(f"  最小值: {min(scores):.3f}")
                    print(f"  最大值: {max(scores):.3f}")

                    # 显示分布
                    count_1 = sum(1 for s in scores if s == 1.0)
                    count_0 = sum(1 for s in scores if s == 0.0)
                    print(f"  分布: 成功={count_1}, 失败={count_0}")
                    print()

            # 按智能体显示统计
            print(f"{'='*80}")
            print(f"[按智能体统计]")
            print(f"{'='*80}\n")

            for agent in ALL_AGENTS:
                if agent in agent_scores:
                    print(f"{agent}:")
                    for scorer_name in ["asr", "tsr", "rr"]:
                        scores = agent_scores[agent][scorer_name]
                        if scores:
                            avg_score = sum(scores) / len(scores)
                            print(f"  {scorer_name.upper()}: {avg_score:.3f} ({len(scores)} 样本)")
                    print()

    print(f"=" * 80)
    print(f"测试完成！")
    print(f"=" * 80)

except Exception as e:
    print(f"\n[错误]")
    print(f"类型: {type(e).__name__}")
    print(f"消息: {str(e)}")

    import traceback
    print(f"\n[详细错误信息]")
    traceback.print_exc()
