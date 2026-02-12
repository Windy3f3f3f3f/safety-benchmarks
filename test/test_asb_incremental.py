"""
ASB 增量测试脚本

功能：
1. 可配置每个智能体的 task 数量和每个 task 的攻击工具数量
2. 从 logs 目录读取已完成的测试结果
3. 自动识别并只运行未完成的样本
4. 计算总体 ASR, RR, TSR

配置示例：
- tasks_per_agent: 2  # 每个智能体 2 个 task
- tools_per_task: 1   # 每个 task 1 个攻击工具
- 总样本数: 10 × 2 × 1 = 20
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

# 修复 Windows 上的 GBK 编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'
import _locale
_locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])

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
print("ASB DPI 攻击 - 增量测试脚本")
print("=" * 80)

# ============ 配置区 ============
# 可修改这些参数来调整测试规模

TASKS_PER_AGENT = 2      # 每个智能体运行多少个 task (0-5)
TOOLS_PER_TASK = 1       # 每个 task 测试多少个攻击工具 (0-20)

# 模型配置
MODEL_NAME = "gpt-4o-mini"
MAX_ITERATIONS = 10
TEMPERATURE = 0.0

# Logs 目录
LOGS_DIR = "logs"

# ============ 配置区结束 ============

# 检查 API 配置
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

print(f"\n[配置]")
print(f"工作目录: {os.getcwd()}")
print(f"模型: {MODEL_NAME}")
print(f"每个智能体 task 数: {TASKS_PER_AGENT}")
print(f"每个 task 攻击工具数: {TOOLS_PER_TASK}")
print(f"目标总样本数: 10 × {TASKS_PER_AGENT} × {TOOLS_PER_TASK} = {10 * TASKS_PER_AGENT * TOOLS_PER_TASK}")

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

# ============ 读取已完成日志 ============
print(f"\n[读取已完成日志]")

def read_completed_samples_from_logs(logs_dir):
    """
    从 logs 目录读取已完成的样本

    使用 inspect_ai 的 EvalLog API 来读取 .eval 文件

    返回: set() 已完成的 sample_id 集合
    """
    completed_samples = set()
    logs_path = Path(logs_dir)

    if not logs_path.exists():
        print(f"  日志目录不存在: {logs_dir}")
        return completed_samples

    # 查找所有 .eval 日志文件
    log_files = list(logs_path.glob("*.eval"))

    print(f"  找到 {len(log_files)} 个 .eval 日志文件")

    if not log_files:
        return completed_samples

    # 使用 inspect_ai 的工具读取日志
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        print(f"  ✗ 无法导入 inspect_ai.log 模块")
        return completed_samples

    for log_file in log_files:
        try:
            # 检查文件是否存在
            if not log_file.exists():
                print(f"  ⊗ {log_file.name}: 文件不存在，跳过")
                continue

            # 使用 inspect_ai 的 API 读取日志
            eval_log = read_eval_log(log_file)

            if eval_log and eval_log.samples:
                sample_count = 0
                for sample in eval_log.samples:
                    if sample.id:
                        completed_samples.add(sample.id)
                        sample_count += 1

                print(f"  ✓ {log_file.name}: 读取到 {sample_count} 个样本")

        except FileNotFoundError:
            print(f"  ⊗ {log_file.name}: 文件不存在，跳过")
        except Exception as e:
            print(f"  ✗ {log_file.name}: 读取失败 - {type(e).__name__}: {e}")

    return completed_samples

completed_sample_ids = read_completed_samples_from_logs(LOGS_DIR)

print(f"\n已完成样本数: {len(completed_sample_ids)}")

# ============ 生成目标样本列表 ============
print(f"\n[生成目标样本列表]")

from eval_benchmarks.asb.dataset import load_asb_dataset
from inspect_ai.dataset import MemoryDataset

# 加载完整数据集
full_dataset = load_asb_dataset(
    agent_names=ALL_AGENTS,
    attack_types=["dpi"],
    limit=None,
    shuffle=False
)

print(f"完整数据集大小: {len(full_dataset.samples)}")

# 生成目标样本列表
target_samples = []
target_sample_ids = set()

for sample in full_dataset.samples:
    agent_name = sample.metadata["agent_name"]
    task_index = sample.metadata["task_index"]
    tool_index = sample.metadata["tool_index"]

    # 筛选条件
    if task_index < TASKS_PER_AGENT and tool_index < TOOLS_PER_TASK:
        target_samples.append(sample)
        target_sample_ids.add(sample.id)

print(f"目标样本数: {len(target_samples)}")

# ============ 筛选未完成样本 ============
print(f"\n[筛选未完成样本]")

pending_samples = []
pending_sample_ids = []

for sample in target_samples:
    if sample.id not in completed_sample_ids:
        pending_samples.append(sample)
        pending_sample_ids.append(sample.id)

print(f"未完成样本数: {len(pending_samples)}")

if pending_samples:
    print(f"\n未完成样本列表:")
    for idx, sample_id in enumerate(pending_sample_ids, 1):
        # 解析 sample_id: agent_taskIndex_toolIndex
        parts = sample_id.split('_')
        agent = '_'.join(parts[:-2])
        task_idx = parts[-2]
        tool_idx = parts[-1]
        print(f"  {idx}. {agent} | task_{task_idx} | tool_{tool_idx}")

# ============ 运行评测 ============
if pending_samples:
    print(f"\n[运行评测]")
    print(f"待测试样本数: {len(pending_samples)}")

    # 创建数据集
    pending_dataset = MemoryDataset(pending_samples)

    # 创建 Task
    from eval_benchmarks.asb.asb import _create_all_tools
    from eval_benchmarks.asb.solver import asb_react_solver
    from eval_benchmarks.asb.scorer import asr_scorer, tsr_scorer, rr_scorer
    from inspect_ai import Task
    from inspect_ai.model import GenerateConfig

    all_tools = _create_all_tools(ALL_AGENTS)

    task = Task(
        dataset=pending_dataset,
        solver=[asb_react_solver(
            max_iterations=MAX_ITERATIONS,
            temperature=TEMPERATURE,
            attack_type="dpi"
        )],
        scorer=[
            asr_scorer(),
            tsr_scorer(),
            rr_scorer()
        ],
        tools=all_tools,
        config=GenerateConfig(
            temperature=TEMPERATURE,
            max_tokens=2048
        )
    )

    from inspect_ai import eval

    try:
        results = eval(
            task,
            model=f"openai/{MODEL_NAME}",
            model_args={
                "api_key": api_key,
            },
            log_limit=len(pending_samples)
        )

        print(f"\n✓ 新样本评测完成")

    except Exception as e:
        print(f"\n✗ 评测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

else:
    print(f"\n[跳过评测]")
    print(f"所有目标样本已完成，无需运行")

# ============ 读取所有结果并计算指标 ============
print(f"\n" + "=" * 80)
print(f"[计算总体指标]")
print(f"=" * 80)

# 重新读取所有日志（包括刚运行的）
all_completed_samples = read_completed_samples_from_logs(LOGS_DIR)

# 收集所有目标样本的评分
all_scores = {
    "asr": [],
    "tsr": [],
    "rr": []
}

agent_scores = defaultdict(lambda: {"asr": [], "tsr": [], "rr": []})

# 使用 inspect_ai 的 API 读取日志文件获取评分
try:
    from inspect_ai.log import read_eval_log
except ImportError:
    print(f"✗ 无法导入 inspect_ai.log 模块")
    sys.exit(1)

logs_path = Path(LOGS_DIR)
log_files = list(logs_path.glob("*.eval"))

for log_file in log_files:
    try:
        # 检查文件是否存在
        if not log_file.exists():
            continue

        eval_log = read_eval_log(log_file)

        if not eval_log or not eval_log.samples:
            continue

        for sample in eval_log.samples:
            sample_id = sample.id

            # 只统计目标样本
            if sample_id not in target_sample_ids:
                continue

            # 解析 agent_name: financial_analyst_agent_0_0
            parts = sample_id.split('_')
            agent_name = '_'.join(parts[:-2])

            # 获取评分
            if sample.scores:
                for metric_name in ["asr", "tsr", "rr"]:
                    if metric_name in sample.scores:
                        score_obj = sample.scores[metric_name]
                        # inspect_ai 的 Score 对象有 value 属性
                        value = score_obj.value if hasattr(score_obj, 'value') else 0.0

                        all_scores[metric_name].append(value)
                        agent_scores[agent_name][metric_name].append(value)

    except Exception as e:
        print(f"读取日志失败 {log_file.name}: {type(e).__name__}: {e}")

# ============ 显示结果 ============
print(f"\n[总体统计]")
print(f"目标样本总数: {len(target_samples)}")
print(f"已完成样本数: {len(all_scores['asr']) if all_scores['asr'] else 0}")
print(f"完成率: {len(all_scores['asr']) / len(target_samples) * 100:.1f}%\n")

if all_scores['asr']:
    for metric_name in ["asr", "tsr", "rr"]:
        scores = all_scores[metric_name]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"{metric_name.upper()}:")
            print(f"  平均值: {avg_score:.3f}")
            print(f"  样本数: {len(scores)}")
            print(f"  最小值: {min(scores):.3f}")
            print(f"  最大值: {max(scores):.3f}")

            # 显示分布
            count_1 = sum(1 for s in scores if s == 1.0)
            count_0 = sum(1 for s in scores if s == 0.0)
            print(f"  分布: 成功={count_1} ({count_1/len(scores)*100:.1f}%), 失败={count_0} ({count_0/len(scores)*100:.1f}%)")
            print()

    # 按智能体统计
    print(f"[按智能体统计]\n")
    for agent in ALL_AGENTS:
        if agent in agent_scores and agent_scores[agent]['asr']:
            print(f"{agent}:")
            for metric_name in ["asr", "tsr", "rr"]:
                scores = agent_scores[agent][metric_name]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"  {metric_name.upper()}: {avg_score:.3f} ({len(scores)} 样本)")
            print()

else:
    print("暂无评分数据")

print(f"=" * 80)
print(f"测试完成！")
print(f"=" * 80)
