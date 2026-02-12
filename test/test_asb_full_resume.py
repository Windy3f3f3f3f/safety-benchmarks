"""
ASB 完整测试脚本 - 支持断点续跑

测试配置:
- 10种智能体
- 每个智能体5个任务
- 每个任务4个攻击工具（Sample）
- 5种攻击方法: dpi, opi, mp, pot, mixed
- 总样本数: 10 × 5 × 4 × 5 = 1000

每种攻击方法的日志分别存储到 logs/{attack_type}/ 目录
支持从已有日志读取已完成样本，实现断点续跑

计算指标:
- ASR (Attack Success Rate) - 攻击成功率
- TSR (Tool Success Rate) - 工具成功率
- RR (Refusal Rate) - 拒绝率
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

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
os.chdir(Path(__file__).parent.parent / "safety-benchmarks")
benchmarks_path = os.path.join(os.getcwd(), "benchmarks")
if benchmarks_path not in sys.path:
    sys.path.insert(0, benchmarks_path)

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# 加载环境变量
from dotenv import load_dotenv
env_path = Path("benchmarks/eval_benchmarks/asb/.env")
load_dotenv(env_path)

# ============================================================================
# 配置参数
# ============================================================================

MODEL_NAME = "gpt-4o"  # 可改为 "gpt-4o-mini" 或其他模型
MAX_ITERATIONS = 10
TEMPERATURE = 0.0
SAMPLES_PER_TASK = 4  # 每个任务取4个攻击工具
TASKS_PER_AGENT = 5   # 每个智能体取5个任务

# 5种攻击方法
ATTACK_TYPES = ["dpi", "opi", "mp", "pot", "mixed"]

# 10个智能体
ALL_AGENTS = [
    # 先只测试一个智能体
    "system_admin_agent",
    # "financial_analyst_agent",
    # "legal_consultant_agent",
    # "medical_advisor_agent",
    # "education_consultant_agent",
    # "psychological_counselor_agent",
    # "ecommerce_manager_agent",
    # "aerospace_engineer_agent",
    # "academic_search_agent",
    # "autonomous_driving_agent",
]

# 日志目录
LOG_BASE_DIR = Path(__file__).parent / "logs"

# ============================================================================
# 断点续跑：从日志文件读取已完成的样本
# ============================================================================

def get_completed_samples_from_logs(attack_type: str) -> set:
    """
    从日志文件读取已完成的样本ID

    参数:
        attack_type: 攻击类型 (dpi, opi, mp, pot, mixed)

    返回:
        已完成样本ID的集合
    """
    completed = set()
    log_dir = LOG_BASE_DIR / attack_type

    if not log_dir.exists():
        return completed

    # 遍历该攻击类型的所有日志文件
    for log_file in log_dir.glob("*.eval"):
        try:
            # .eval 文件是 zip 格式，包含 header.json 和其他 JSON 文件
            with zipfile.ZipFile(log_file, 'r') as zf:
                # 读取 header.json
                if 'header.json' in zf.namelist():
                    with zf.open('header.json') as f:
                        header = json.load(f)

                    # 从 header 中获取样本信息
                    if 'samples' in header:
                        for sample in header['samples']:
                            if 'id' in sample:
                                completed.add(sample['id'])

        except Exception as e:
            print(f"  [警告] 读取日志文件 {log_file.name} 失败: {e}")
            continue

    return completed


def parse_eval_log(log_path: Path) -> dict:
    """
    解析 .eval 日志文件，提取样本结果

    返回:
        {
            "sample_id": {
                "asr": float,
                "tsr": float,
                "rr": float,
                ...
            },
            ...
        }
    """
    results = {}

    try:
        with zipfile.ZipFile(log_path, 'r') as zf:
            if 'header.json' in zf.namelist():
                with zf.open('header.json') as f:
                    header = json.load(f)

                if 'samples' in header:
                    for sample in header['samples']:
                        sample_id = sample.get('id', '')
                        if not sample_id:
                            continue

                        scores = {}
                        if 'scores' in sample:
                            for score in sample['scores']:
                                scorer_name = score.get('scorer', '')
                                value = score.get('value', 0)
                                scores[scorer_name] = value

                        results[sample_id] = scores

    except Exception as e:
        print(f"  [警告] 解析日志 {log_path.name} 失败: {e}")

    return results


def load_previous_results(attack_type: str) -> dict:
    """
    加载该攻击类型的所有历史结果

    返回:
        {
            "sample_id": {"asr": x, "tsr": y, "rr": z},
            ...
        }
    """
    all_results = {}
    log_dir = LOG_BASE_DIR / attack_type

    if not log_dir.exists():
        return all_results

    for log_file in log_dir.glob("*.eval"):
        results = parse_eval_log(log_file)
        all_results.update(results)

    return all_results

# ============================================================================
# 主测试逻辑
# ============================================================================

def run_attack_type_test(attack_type: str, api_key: str, base_url: str):
    """
    运行单个攻击类型的测试

    参数:
        attack_type: 攻击类型
        api_key: API密钥
        base_url: API基础URL
    """
    print(f"\n{'='*80}")
    print(f"攻击类型: {attack_type.upper()}")
    print(f"{'='*80}")

    # 创建日志目录
    log_dir = LOG_BASE_DIR / attack_type
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取已完成的样本
    print(f"\n[1] 检查已完成样本...")
    completed_samples = get_completed_samples_from_logs(attack_type)
    print(f"  已完成样本数: {len(completed_samples)}")

    # 2. 加载历史结果
    print(f"\n[2] 加载历史结果...")
    previous_results = load_previous_results(attack_type)
    print(f"  历史结果数: {len(previous_results)}")

    # 3. 加载数据集并筛选
    print(f"\n[3] 加载数据集...")
    from eval_benchmarks.asb.dataset import load_asb_dataset
    from inspect_ai.dataset import MemoryDataset, Sample

    full_dataset = load_asb_dataset(
        agent_names=ALL_AGENTS,
        attack_types=[attack_type],
        limit=None,
        shuffle=False
    )

    print(f"  完整数据集大小: {len(full_dataset.samples)}")

    # 筛选样本: 每个智能体取前5个task，每个task取前4个攻击工具
    print(f"\n[4] 筛选样本...")
    print(f"  筛选规则: 每个智能体 {TASKS_PER_AGENT} 个任务，每个任务 {SAMPLES_PER_TASK} 个攻击工具")

    filtered_samples = []
    sample_stats = defaultdict(lambda: {"tasks": set(), "tools": defaultdict(int)})

    for sample in full_dataset.samples:
        agent_name = sample.metadata["agent_name"]
        task_index = sample.metadata["task_index"]
        tool_index = sample.metadata["tool_index"]

        # 筛选条件
        if task_index < TASKS_PER_AGENT and tool_index < SAMPLES_PER_TASK:
            sample_id = sample.id

            # 检查是否已完成
            if sample_id in completed_samples:
                print(f"  [跳过] {sample_id} (已完成)")
                continue

            filtered_samples.append(sample)
            sample_stats[agent_name]["tasks"].add(task_index)
            sample_stats[agent_name]["tools"][task_index] += 1

    print(f"\n  筛选后待测试样本数: {len(filtered_samples)}")

    if len(filtered_samples) == 0:
        print(f"\n  所有样本已完成，跳过测试")
        return previous_results

    # 显示各智能体筛选统计
    print(f"\n  各智能体待测试样本:")
    for agent in ALL_AGENTS:
        stats = sample_stats[agent]
        total = sum(stats["tools"].values())
        if total > 0:
            print(f"    {agent}: {total} 个样本")

    # 4. 创建 Task
    print(f"\n[5] 创建 Task...")
    from eval_benchmarks.asb.asb import _create_all_tools
    from eval_benchmarks.asb.solver import asb_react_solver
    from eval_benchmarks.asb.scorer import asr_scorer, tsr_scorer, rr_scorer
    from inspect_ai import Task
    from inspect_ai.model import GenerateConfig

    # 创建所有工具
    all_tools = _create_all_tools(ALL_AGENTS)

    # 创建 Task
    task = Task(
        dataset=MemoryDataset(filtered_samples),
        solver=[asb_react_solver(
            max_iterations=MAX_ITERATIONS,
            temperature=TEMPERATURE,
            attack_type=attack_type
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

    print(f"  Task 数据集大小: {len(task.dataset)}")

    # 5. 运行测试
    print(f"\n[6] 运行评测...")
    print(f"  模型: openai/{MODEL_NAME}")
    print(f"  温度: {TEMPERATURE}")
    print(f"  最大迭代: {MAX_ITERATIONS}")
    print(f"  日志目录: {log_dir}")

    from inspect_ai import eval

    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_file = log_dir / f"{timestamp}_{attack_type}"

    try:
        results = eval(
            task,
            model=f"openai/{MODEL_NAME}",
            model_args={
                "api_key": api_key,
            },
            log_dir=str(log_dir),
            log_format="eval"
        )

        # 6. 收集结果
        print(f"\n[7] 收集结果...")

        if results and len(results) > 0:
            log = results[0]

            if log.samples:
                for sample in log.samples:
                    sample_id = sample.id
                    if hasattr(sample, 'scores') and sample.scores:
                        scores = {}
                        for scorer_name in ["asr", "tsr", "rr"]:
                            if scorer_name in sample.scores:
                                scores[scorer_name] = sample.scores[scorer_name].value
                        previous_results[sample_id] = scores

        print(f"  当前结果总数: {len(previous_results)}")

    except Exception as e:
        print(f"\n[错误] 测试失败: {e}")
        import traceback
        traceback.print_exc()

    return previous_results


def calculate_statistics(results: dict) -> dict:
    """
    计算统计指标

    参数:
        results: {"sample_id": {"asr": x, "tsr": y, "rr": z}, ...}

    返回:
        {
            "asr": {"avg": x, "min": y, "max": z, "count": n},
            "tsr": {...},
            "rr": {...}
        }
    """
    stats = {}

    for metric in ["asr", "tsr", "rr"]:
        values = [r.get(metric, 0) for r in results.values() if metric in r]
        if values:
            stats[metric] = {
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
                "success_count": sum(1 for v in values if v == 1.0),
                "fail_count": sum(1 for v in values if v == 0.0)
            }
        else:
            stats[metric] = {"avg": 0, "min": 0, "max": 0, "count": 0, "success_count": 0, "fail_count": 0}

    return stats


def print_summary(all_results: dict):
    """
    打印总体结果汇总

    参数:
        all_results: {attack_type: {sample_id: {scores}}}
    """
    print(f"\n{'='*80}")
    print(f"总体结果汇总")
    print(f"{'='*80}")

    # 按攻击类型汇总
    print(f"\n按攻击类型统计:")
    print(f"{'-'*80}")
    print(f"{'攻击类型':<10} {'样本数':<10} {'ASR':<15} {'TSR':<15} {'RR':<15}")
    print(f"{'-'*80}")

    for attack_type in ATTACK_TYPES:
        if attack_type in all_results:
            results = all_results[attack_type]
            stats = calculate_statistics(results)

            asr = stats["asr"]
            tsr = stats["tsr"]
            rr = stats["rr"]

            print(f"{attack_type:<10} {asr['count']:<10} "
                  f"{asr['avg']:.3f} ({asr['success_count']}/{asr['count']})    "
                  f"{tsr['avg']:.3f} ({tsr['success_count']}/{tsr['count']})    "
                  f"{rr['avg']:.3f} ({rr['success_count']}/{rr['count']})")

    # 总体统计
    print(f"\n{'-'*80}")
    all_samples = {}
    for attack_type, results in all_results.items():
        all_samples.update(results)

    if all_samples:
        total_stats = calculate_statistics(all_samples)
        print(f"{'总计':<10} {total_stats['asr']['count']:<10} "
              f"{total_stats['asr']['avg']:.3f} ({total_stats['asr']['success_count']}/{total_stats['asr']['count']})    "
              f"{total_stats['tsr']['avg']:.3f} ({total_stats['tsr']['success_count']}/{total_stats['tsr']['count']})    "
              f"{total_stats['rr']['avg']:.3f} ({total_stats['rr']['success_count']}/{total_stats['rr']['count']})")

    print(f"{'='*80}")


def save_results_to_json(all_results: dict, output_path: Path):
    """
    保存结果到 JSON 文件
    """
    output_data = {
        "config": {
            "model": MODEL_NAME,
            "agents": ALL_AGENTS,
            "attack_types": ATTACK_TYPES,
            "tasks_per_agent": TASKS_PER_AGENT,
            "samples_per_task": SAMPLES_PER_TASK,
            "timestamp": datetime.now().isoformat()
        },
        "results": {}
    }

    for attack_type in ATTACK_TYPES:
        if attack_type in all_results:
            results = all_results[attack_type]
            stats = calculate_statistics(results)

            output_data["results"][attack_type] = {
                "statistics": stats,
                "samples": results
            }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 80)
    print("ASB 完整测试 - 支持断点续跑")
    print("=" * 80)

    # 检查 API 配置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    print(f"\n[配置检查]")
    print(f"工作目录: {os.getcwd()}")
    print(f"API Key: {api_key[:20]}...{api_key[-10:] if api_key else 'None'}")
    print(f"Base URL: {base_url}")
    print(f"模型: {MODEL_NAME}")
    print(f"攻击类型: {ATTACK_TYPES}")
    print(f"智能体数量: {len(ALL_AGENTS)}")
    print(f"每智能体任务数: {TASKS_PER_AGENT}")
    print(f"每任务样本数: {SAMPLES_PER_TASK}")
    print(f"预期总样本数: {len(ALL_AGENTS) * TASKS_PER_AGENT * SAMPLES_PER_TASK * len(ATTACK_TYPES)}")
    print(f"日志目录: {LOG_BASE_DIR}")

    print(f"\n[智能体列表]")
    for i, agent in enumerate(ALL_AGENTS, 1):
        print(f"  {i}. {agent}")

    # 存储所有攻击类型的结果
    all_results = {}

    # 遍历每种攻击类型
    for attack_type in ATTACK_TYPES:
        results = run_attack_type_test(attack_type, api_key, base_url)
        all_results[attack_type] = results

        # 打印当前攻击类型的统计
        stats = calculate_statistics(results)
        print(f"\n[{attack_type.upper()}] 测试完成")
        print(f"  样本数: {stats['asr']['count']}")
        print(f"  ASR: {stats['asr']['avg']:.3f}")
        print(f"  TSR: {stats['tsr']['avg']:.3f}")
        print(f"  RR: {stats['rr']['avg']:.3f}")

    # 打印总体汇总
    print_summary(all_results)

    # 保存结果到 JSON
    output_path = LOG_BASE_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results_to_json(all_results, output_path)

    print(f"\n{'='*80}")
    print(f"所有测试完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
