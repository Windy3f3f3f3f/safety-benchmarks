#!/usr/bin/env python3
"""
统一 Eval 入口脚本 - 每个 benchmark 使用独立虚拟环境

用法:
    ./run-eval.py <benchmark>[:<task>] --model <model_name> [options]

示例:
    ./run-eval.py strong_reject --model doubao-seed-1-8-251228
    ./run-eval.py cyberseceval_2:cyse2_interpreter_abuse --model doubao-seed-1-8-251228
    ./run-eval.py truthfulqa --model doubao-seed-1-8-251228 --limit 10

一键测评:
    ./run-eval.py --run-all --model <model_name>    # 运行所有 benchmark
    ./run-eval.py --preflight                        # 仅运行预检查

环境管理:
    ./run-eval.py --setup <benchmark>    # 仅设置环境，不运行
    ./run-eval.py --setup-all            # 设置所有 benchmark 环境
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# 导入预检查模块
sys.path.insert(0, str(Path(__file__).parent / "benchmarks"))
from preflight import (
    JudgeModelConfig,
    run_preflight_checks,
    print_preflight_report,
    get_required_permissions,
)


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.resolve()
VENVS_DIR = PROJECT_ROOT / ".venvs"
UPSTREAM_DIR = PROJECT_ROOT / "upstream"
LOCAL_BENCH_DIR = PROJECT_ROOT / "benchmarks" / "eval_benchmarks"
INDEXES_DIR = PROJECT_ROOT / "benchmarks" / "indexes"
TOOLS_DIR = PROJECT_ROOT / "benchmarks" / "tools"


def load_catalog():
    """加载 benchmark 路由配置"""
    catalog_path = PROJECT_ROOT / "benchmarks" / "catalog.yaml"
    with open(catalog_path, "r") as f:
        return yaml.safe_load(f)


def sanitize_model_name(model_name: str) -> str:
    """处理模型名称：斜杠替换为下划线"""
    return model_name.replace("/", "_")


def normalize_model_name(model_name: str) -> str:
    """
    规范化模型名称为 inspect_ai 所需格式。
    如果模型名称不包含 '/'，则默认添加 'openai/' 前缀。
    """
    if "/" not in model_name:
        return f"openai/{model_name}"
    return model_name


def get_venv_path(benchmark_name: str) -> Path:
    """获取 benchmark 的虚拟环境路径"""
    return VENVS_DIR / benchmark_name


def get_venv_python(benchmark_name: str) -> Path:
    """获取 benchmark 虚拟环境的 Python 路径"""
    venv_dir = "Scripts" if platform.system() == "Windows" else "bin"
    python_exe = "python.exe" if platform.system() == "Windows" else "python"
    return get_venv_path(benchmark_name) / venv_dir / python_exe


def get_venv_inspect(benchmark_name: str) -> Path:
    """获取 benchmark 虚拟环境的 inspect 命令路径"""
    venv_dir = "Scripts" if platform.system() == "Windows" else "bin"
    return get_venv_path(benchmark_name) / venv_dir / "inspect"


def setup_benchmark_env(benchmark_name: str, config: dict, force: bool = False) -> bool:
    """
    为 benchmark 设置独立虚拟环境

    返回 True 表示成功，False 表示失败
    """
    venv_path = get_venv_path(benchmark_name)
    python_version = config.get("python", "3.10")
    extras = config.get("extras", [])
    source = config.get("source", "upstream")

    # 检查是否已存在
    if venv_path.exists() and not force:
        inspect_path = get_venv_inspect(benchmark_name)
        if inspect_path.exists():
            print(f"  环境已存在: {venv_path}")
            return True

    print(f"  创建环境: {venv_path} (Python {python_version})")

    # 创建虚拟环境
    uv_venv_cmd = ["uv", "venv", str(venv_path), "--python", python_version]
    if force:
        uv_venv_cmd.append("--clear")

    result = subprocess.run(
        uv_venv_cmd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  错误: 创建虚拟环境失败")
        print(result.stderr)
        return False

    # 安装 inspect_ai
    print(f"  安装 inspect_ai...")
    result = subprocess.run(
        ["uv", "pip", "install", "-p", str(venv_path),
         "-e", str(UPSTREAM_DIR / "inspect_ai")],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  错误: 安装 inspect_ai 失败")
        print(result.stderr)
        return False

    # 检查 upstream/inspect_evals 子模块是否已初始化
    inspect_evals_dir = UPSTREAM_DIR / "inspect_evals"
    if not (inspect_evals_dir / "pyproject.toml").exists():
        print(f"  错误: upstream/inspect_evals 未初始化。请运行:")
        print(f"    git submodule update --init --recursive")
        return False

    # 始终安装 inspect_evals（所有 benchmark 都需要）
    install_spec = str(inspect_evals_dir)
    if extras:
        extras_str = ",".join(extras)
        install_spec = f"{install_spec}[{extras_str}]"

    extras_display = f"[{','.join(extras)}]" if extras else ""
    print(f"  安装 inspect_evals{extras_display}...")
    result = subprocess.run(
        ["uv", "pip", "install", "-p", str(venv_path), "-e", install_spec],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  错误: 安装 inspect_evals 失败")
        print(result.stderr)
        return False

    # Local benchmark: 额外安装 eval_benchmarks 包
    if source == "local":
        print(f"  安装 eval_benchmarks package...")
        benchmarks_dir = PROJECT_ROOT / "benchmarks"
        result = subprocess.run(
            ["uv", "pip", "install", "-p", str(venv_path), "-e", str(benchmarks_dir)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"  错误: 安装 eval_benchmarks package 失败")
            print(result.stderr)
            return False

        # 安装 benchmark 特有依赖 (requirements.txt)
        module_name = config.get("module", "").split("/")[-1]
        local_benchmark_dir = LOCAL_BENCH_DIR / module_name
        requirements_file = local_benchmark_dir / "requirements.txt"

        if requirements_file.exists():
            print(f"  安装 {module_name} 依赖...")
            result = subprocess.run(
                ["uv", "pip", "install", "-p", str(venv_path), "-r", str(requirements_file)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"  警告: 安装 {module_name} 依赖失败")
                print(result.stderr)

    # 安装 openai (必需)
    print(f"  安装 openai...")
    result = subprocess.run(
        ["uv", "pip", "install", "-p", str(venv_path), "openai"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"  错误: 安装 openai 失败")
        print(result.stderr)
        return False

    # cve_bench 需要单独安装 cvebench 包（uv extras 处理有问题）
    # 新版 cvebench 已包含所有 challenges 和 docker 配置数据
    if benchmark_name == "cve_bench":
        print(f"  安装 cvebench...")
        result = subprocess.run(
            ["uv", "pip", "install", "-p", str(venv_path),
             "cvebench @ git+https://github.com/Scott-Simmons/cve-bench.git@92541add2ebd89e5b15ed260eb5d0e9b5102c33e"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"  错误: 安装 cvebench 失败")
            print(result.stderr)
            return False

    print(f"  环境设置完成")
    return True


def get_index_path(benchmark_name: str, task_name: str) -> Path:
    """获取索引文件路径"""
    return INDEXES_DIR / benchmark_name / f"{task_name}.yaml"


def expand_sample_ranges(samples: list[str]) -> list[str]:
    """
    展开范围语法，如 "1-10" -> ["1", "2", ..., "10"]

    支持格式:
    - "5"      -> ["5"]
    - "1-10"   -> ["1", "2", ..., "10"]
    - "sample-*"  -> 保持原样 (通配符)
    """
    result = []
    for s in samples:
        # 跳过通配符
        if "*" in s or "?" in s:
            result.append(s)
            continue

        # 尝试解析范围语法
        match = re.match(r"^(\d+)-(\d+)$", s)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            result.extend(str(i) for i in range(start, end + 1))
        else:
            result.append(s)
    return result


def load_index_file(index_path: Path) -> tuple[str, list[str]] | None:
    """
    加载索引文件 (兼容新旧格式)

    返回 (mode, sample_ids) 或 None (如果文件不存在)
    mode: "include" 或 "exclude"

    支持两种格式:
    - 旧格式: samples 是 list ["1", "2-5", ...]
    - 新格式: samples 是 dict {"1": {sources, added}, ...}
    """
    if not index_path.exists():
        return None

    with open(index_path, "r") as f:
        data = yaml.safe_load(f)

    if not data:
        return None

    mode = data.get("mode", "include")
    samples_data = data.get("samples", [])

    if not samples_data:
        return None

    # 新格式: dict {id: {sources, added}}
    if isinstance(samples_data, dict):
        # keys 可能包含范围语法 (旧格式迁移后)，需要展开
        sample_ids = expand_sample_ranges(list(samples_data.keys()))
    # 旧格式: list ["1", "2-5", ...]
    elif isinstance(samples_data, list):
        sample_ids = expand_sample_ranges(samples_data)
    else:
        return None

    return mode, sample_ids


def match_sample_id(sample_id: str, patterns: list[str]) -> bool:
    """检查样本 ID 是否匹配任一模式"""
    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            if fnmatch.fnmatch(sample_id, pattern):
                return True
        elif sample_id == pattern:
            return True
    return False


def list_samples(benchmark_name: str, task_spec: str) -> list[str]:
    """
    调用辅助脚本获取样本 ID 列表

    在 benchmark 的虚拟环境中运行 list_samples.py
    """
    python_path = get_venv_python(benchmark_name)
    list_samples_script = TOOLS_DIR / "list_samples.py"

    if not python_path.exists():
        raise RuntimeError(f"虚拟环境不存在: {benchmark_name}，请先运行 --setup")

    result = subprocess.run(
        [str(python_path), str(list_samples_script), task_spec],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"获取样本列表失败: {stderr}")

    try:
        data = json.loads(result.stdout)
        if "error" in data:
            raise RuntimeError(data["error"])
        return data["ids"]
    except json.JSONDecodeError as e:
        raise RuntimeError(f"解析样本列表失败: {e}")


def generate_index_file(
    benchmark_name: str,
    task_name: str,
    task_spec: str,
    output_path: Path | None = None,
) -> Path:
    """
    生成初始索引文件

    包含所有样本 ID，默认 mode 为 include
    """
    sample_ids = list_samples(benchmark_name, task_spec)

    if output_path is None:
        output_path = get_index_path(benchmark_name, task_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"""# {benchmark_name}:{task_name} 样本索引
# 生成时间: {datetime.now().strftime("%Y-%m-%d")}
# 总样本数: {len(sample_ids)}
#
# mode: include = 只跑列出的样本
# mode: exclude = 跳过列出的样本
#
# 支持语法:
#   - "5"       单个样本
#   - "1-10"    范围 (展开为 1, 2, ..., 10)
#   - "sample-*" 通配符

"""

    content = {
        "mode": "include",
        "samples": sample_ids,
    }

    with open(output_path, "w") as f:
        f.write(header)
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True)

    return output_path


def resolve_task(benchmark_spec: str, catalog: dict) -> tuple[str, str, dict, dict]:
    """
    解析 benchmark 规格，返回 (benchmark_name, task_spec, config, task_config)
    """
    if ":" in benchmark_spec:
        benchmark_name, task_name = benchmark_spec.split(":", 1)
    else:
        benchmark_name = benchmark_spec
        task_name = None

    benchmarks = catalog.get("benchmarks", {})
    if benchmark_name not in benchmarks:
        available = ", ".join(benchmarks.keys())
        print(f"错误: 未知的 benchmark '{benchmark_name}'")
        print(f"可用的 benchmarks: {available}")
        sys.exit(1)

    config = benchmarks[benchmark_name]
    tasks = config.get("tasks", [])

    if not tasks:
        print(f"错误: benchmark '{benchmark_name}' 没有定义任何 task")
        sys.exit(1)

    def find_task(name: str):
        for t in tasks:
            if isinstance(t, dict):
                if t.get("name") == name:
                    return t
            elif t == name:
                return {"name": t, "path": config["module"]}
        return None

    if task_name:
        task = find_task(task_name)
        if not task:
            available = ", ".join(
                t.get("name") if isinstance(t, dict) else t
                for t in tasks
            )
            print(f"错误: benchmark '{benchmark_name}' 中没有 task '{task_name}'")
            print(f"可用的 tasks: {available}")
            sys.exit(1)
    else:
        first_task = tasks[0]
        if isinstance(first_task, dict):
            task = first_task
        else:
            task = {"name": first_task, "path": config["module"]}

    task_spec = task["path"]
    return benchmark_name, task_spec, config, task


def run_eval(benchmark_name: str, task_spec: str, config: dict,
             model: str, limit: int = None, judge_model: str = None,
             extra_args: list = None, dry_run: bool = False,
             task_config: dict = None,
             no_index: bool = False, index_file: Path = None,
             api_base: str = None, api_key: str = None,
             log_suffix: str = None) -> int:
    """运行评估"""

    # 确保环境存在
    inspect_path = get_venv_inspect(benchmark_name)
    if not inspect_path.exists():
        print(f"设置 {benchmark_name} 环境...")
        if not setup_benchmark_env(benchmark_name, config):
            return 1

    # 规范化模型名称
    model_for_inspect = normalize_model_name(model)

    # 设置结果目录
    # mm_safety_bench 使用单独目录（按样本数）
    if benchmark_name == "mm_safety_bench":
        # 对于limit参数，使用具体数字；否则用"all"
        sample_count = str(limit) if limit else "all"
        # 支持 --log-suffix 参数（例如: 2 -> total2_100）
        dir_prefix = f"total{log_suffix}" if log_suffix else "total"
        results_dir = PROJECT_ROOT / "logs" / f"{dir_prefix}_{sample_count}"
    else:
        results_dir = PROJECT_ROOT / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 设置环境变量
    env = os.environ.copy()
    env["INSPECT_LOG_DIR"] = str(results_dir)

    # 清除可能影响 inspect_ai 缓存路径的 VSCode 扩展环境变量
    for key in ["INSPECT_WORKSPACE_ID", "INSPECT_VSCODE_EXT_VERSION"]:
        env.pop(key, None)

    # 确定 judge model
    effective_judge = judge_model or config.get("judge_model")
    if effective_judge:
        effective_judge = normalize_model_name(effective_judge)

    # 处理索引文件
    sample_ids = None
    index_mode = None
    if not no_index:
        # 确定索引文件路径
        if index_file:
            idx_path = index_file
        else:
            task_name = task_config.get("name") if task_config else None
            if task_name:
                idx_path = get_index_path(benchmark_name, task_name)
            else:
                idx_path = None

        # 加载索引文件
        if idx_path:
            index_data = load_index_file(idx_path)
            if index_data:
                index_mode, sample_ids = index_data
                print(f"Index file: {idx_path}")
                print(f"Index mode: {index_mode}, {len(sample_ids)} samples")

    # 如果显式传了 api_key，覆盖子进程环境变量
    if api_key:
        env["OPENAI_API_KEY"] = api_key

    # 构建命令
    cmd = [str(inspect_path), "eval", task_spec, "--model", model_for_inspect]

    # 显式传递 model base URL（优先级高于 .env 中的 OPENAI_BASE_URL）
    if api_base:
        cmd.extend(["--model-base-url", api_base])

    # 添加样本 ID 过滤 (仅支持 include 模式)
    # 注意: inspect_ai 不允许同时指定 --sample-id 和 --limit
    has_sample_ids = False
    if sample_ids and index_mode == "include":
        # inspect_ai --sample-id 接受逗号分隔的 ID 列表
        # 需要过滤掉通配符模式 (inspect_ai 不支持)
        literal_ids = [sid for sid in sample_ids if "*" not in sid and "?" not in sid]
        if literal_ids:
            # 如果同时指定了 limit，截取前 N 个 sample_id 而非传 --limit
            if limit and len(literal_ids) > limit:
                literal_ids = literal_ids[:limit]
                print(f"Index + limit: 从 {len(sample_ids)} 个样本中取前 {limit} 个")
            cmd.extend(["--sample-id", ",".join(literal_ids)])
            has_sample_ids = True
    elif sample_ids and index_mode == "exclude":
        # exclude 模式需要先获取所有样本 ID，然后排除
        print("警告: exclude 模式需要先获取完整样本列表，这可能较慢...")
        try:
            all_ids = list_samples(benchmark_name, task_spec)
            included_ids = [
                sid for sid in all_ids
                if not match_sample_id(sid, sample_ids)
            ]
            if included_ids:
                if limit and len(included_ids) > limit:
                    included_ids = included_ids[:limit]
                    print(f"Exclude + limit: 取前 {limit} 个样本")
                cmd.extend(["--sample-id", ",".join(included_ids)])
                has_sample_ids = True
                print(f"Exclude 后剩余样本数: {len(included_ids)}")
        except Exception as e:
            print(f"警告: 获取样本列表失败，跳过索引过滤: {e}")

    if limit and not has_sample_ids:
        cmd.extend(["--limit", str(limit)])

    if effective_judge:
        cmd.extend(["--model-role", f"grader={effective_judge}"])
        # 通过 -T 直接覆盖 task 函数的 judge model 参数
        # (解决 task 硬编码默认值导致 --model-role 被忽略的问题)
        judge_param = config.get("judge_param")
        if judge_param:
            cmd.extend(["-T", f"{judge_param}={effective_judge}"])

    # 添加 task_args (来自 catalog.yaml)
    if task_config:
        task_args = task_config.get("task_args", {})
        for key, value in task_args.items():
            cmd.extend(["-T", f"{key}={value}"])

    if extra_args:
        cmd.extend(extra_args)

    # 打印信息
    print(f"Benchmark: {benchmark_name}")
    print(f"Task: {task_spec}")
    print(f"Model: {model_for_inspect}")
    print(f"Results dir: {results_dir}")
    if effective_judge:
        print(f"Judge model: {effective_judge}")
    print(f"Command: {' '.join(cmd)}")
    print()

    if dry_run:
        print("[Dry run - 不实际执行]")
        return 0

    # 执行命令
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="统一 Eval 入口脚本 (独立虚拟环境)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "benchmark",
        nargs="?",
        help="Benchmark 名称，格式: benchmark_name 或 benchmark_name:task_name"
    )
    parser.add_argument(
        "--model", "-m",
        help="模型名称 (例如: doubao-seed-1-8-251228)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="限制样本数量"
    )
    parser.add_argument(
        "--judge-model",
        help="Judge 模型 (覆盖 catalog 中的默认值)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印命令，不实际执行"
    )
    parser.add_argument(
        "--setup",
        metavar="BENCHMARK",
        help="仅设置指定 benchmark 的环境"
    )
    parser.add_argument(
        "--setup-all",
        action="store_true",
        help="设置所有 benchmark 的环境"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新创建环境"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="运行预检查，检测环境依赖"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="运行所有 benchmark (一键测评)"
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过预检查 (不推荐)"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="自动确认权限 (用于非交互式运行)"
    )
    # 索引相关参数
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="禁用索引，跑全量样本"
    )
    parser.add_argument(
        "--index-file",
        type=Path,
        help="指定索引文件路径"
    )
    parser.add_argument(
        "--generate-index",
        action="store_true",
        help="生成初始索引文件（不运行评测）"
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="列出所有样本 ID（不运行评测）"
    )
    parser.add_argument(
        "--api-base",
        help="模型 API 的 Base URL（覆盖 .env 中的 OPENAI_BASE_URL）"
    )
    parser.add_argument(
        "--api-key",
        help="模型 API Key（覆盖 .env 中的 OPENAI_API_KEY）"
    )
    parser.add_argument(
        "--log-suffix",
        help="日志目录后缀（例如: 2 会生成 total2_100 而非 total_100）"
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="传递给 inspect eval 的额外参数"
    )

    args = parser.parse_args()
    catalog = load_catalog()
    benchmarks = catalog.get("benchmarks", {})

    # 仅运行预检查
    if args.preflight:
        benchmark_list = list(benchmarks.keys())
        judge_config = JudgeModelConfig(
            model_name=args.judge_model or catalog.get("benchmarks", {}).get("cyberseceval_2", {}).get("judge_model", "")
        )
        results = run_preflight_checks(benchmark_list, judge_config)
        passed = print_preflight_report(results)

        # 显示权限确认
        permissions = get_required_permissions(benchmark_list)
        if permissions:
            print("\n需要用户确认的权限:")
            for perm in permissions:
                print(f"  • {perm}")
            print()

        return 0 if passed else 1

    # 一键测评
    if args.run_all:
        if not args.model:
            print("错误: --run-all 必须指定 --model 参数")
            return 1

        benchmark_list = list(benchmarks.keys())

        # 预检查
        if not args.skip_preflight:
            judge_config = JudgeModelConfig(
                model_name=args.judge_model or benchmarks.get("cyberseceval_2", {}).get("judge_model", "")
            )
            results = run_preflight_checks(benchmark_list, judge_config)
            passed = print_preflight_report(results)

            # 显示并确认权限
            permissions = get_required_permissions(benchmark_list)
            if permissions:
                print("\n需要用户确认的权限:")
                for perm in permissions:
                    print(f"  • {perm}")
                print()

                if not args.confirm:
                    try:
                        response = input("是否同意上述权限? [y/N]: ").strip().lower()
                        if response not in ("y", "yes"):
                            print("用户取消")
                            return 1
                    except (EOFError, KeyboardInterrupt):
                        print("\n用户取消")
                        return 1

            if not passed and not args.skip_preflight:
                print("\n预检查未通过。使用 --skip-preflight 可跳过检查（不推荐）")
                return 1

        # 运行所有 benchmark 的所有 tasks
        total_tasks = sum(
            len(config.get("tasks", [])) for config in benchmarks.values()
        )
        print(f"\n开始一键测评: {args.model}")
        print(f"共 {len(benchmarks)} 个 benchmark, {total_tasks} 个 tasks\n")
        print("=" * 60)

        VENVS_DIR.mkdir(parents=True, exist_ok=True)

        results_summary = []
        for name, config in benchmarks.items():
            tasks = config.get("tasks", [])
            if not tasks:
                print(f"\n[{name}]")
                print(f"  跳过: 没有定义 task")
                results_summary.append((name, None, "skipped", "no tasks"))
                continue

            # 遍历每个 task
            for task in tasks:
                if isinstance(task, dict):
                    task_name = task.get("name")
                    task_spec = task["path"]
                    task_config = task
                else:
                    task_name = task
                    task_spec = config["module"]
                    task_config = {"name": task, "path": task_spec}

                print(f"\n[{name}:{task_name}]")

                returncode = run_eval(
                    benchmark_name=name,
                    task_spec=task_spec,
                    config=config,
                    model=args.model,
                    limit=args.limit,
                    judge_model=args.judge_model,
                    extra_args=args.extra_args,
                    dry_run=args.dry_run,
                    task_config=task_config,
                    no_index=args.no_index,
                    index_file=args.index_file,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    log_suffix=args.log_suffix,
                )

                if returncode == 0:
                    results_summary.append((name, task_name, "success", ""))
                else:
                    results_summary.append((name, task_name, "failed", f"exit code {returncode}"))

        # 打印汇总
        print("\n" + "=" * 60)
        print("测评结果汇总")
        print("=" * 60)
        success_count = sum(1 for _, _, status, _ in results_summary if status == "success")
        for name, task_name, status, msg in results_summary:
            icon = "✅" if status == "success" else "❌" if status == "failed" else "⏭️"
            display_name = f"{name}:{task_name}" if task_name else name
            print(f"  {icon} {display_name}: {status} {msg}")

        print(f"\n通过: {success_count}/{len(results_summary)}")
        return 0 if success_count == len(results_summary) else 1

    # 设置所有环境
    if args.setup_all:
        print("设置所有 benchmark 环境...")
        VENVS_DIR.mkdir(parents=True, exist_ok=True)
        success = True
        for name, config in benchmarks.items():
            print(f"\n[{name}]")
            if not setup_benchmark_env(name, config, args.force):
                success = False
        return 0 if success else 1

    # 设置单个环境
    if args.setup:
        if args.setup not in benchmarks:
            print(f"错误: 未知的 benchmark '{args.setup}'")
            return 1
        print(f"设置 {args.setup} 环境...")
        VENVS_DIR.mkdir(parents=True, exist_ok=True)
        return 0 if setup_benchmark_env(args.setup, benchmarks[args.setup], args.force) else 1

    # 列出样本 ID
    if args.list_samples:
        if not args.benchmark:
            print("错误: --list-samples 必须指定 benchmark")
            return 1

        benchmark_name, task_spec, config, task_config = resolve_task(args.benchmark, catalog)

        # 确保环境存在
        VENVS_DIR.mkdir(parents=True, exist_ok=True)
        inspect_path = get_venv_inspect(benchmark_name)
        if not inspect_path.exists():
            print(f"设置 {benchmark_name} 环境...")
            if not setup_benchmark_env(benchmark_name, config):
                return 1

        try:
            sample_ids = list_samples(benchmark_name, task_spec)
            print(f"Task: {args.benchmark}")
            print(f"Total samples: {len(sample_ids)}")
            print()
            for sid in sample_ids:
                print(sid)
            return 0
        except Exception as e:
            print(f"错误: {e}")
            return 1

    # 生成索引文件
    if args.generate_index:
        if not args.benchmark:
            print("错误: --generate-index 必须指定 benchmark")
            return 1

        benchmark_name, task_spec, config, task_config = resolve_task(args.benchmark, catalog)
        task_name = task_config.get("name")

        # 确保环境存在
        VENVS_DIR.mkdir(parents=True, exist_ok=True)
        inspect_path = get_venv_inspect(benchmark_name)
        if not inspect_path.exists():
            print(f"设置 {benchmark_name} 环境...")
            if not setup_benchmark_env(benchmark_name, config):
                return 1

        try:
            output_path = args.index_file or get_index_path(benchmark_name, task_name)
            path = generate_index_file(benchmark_name, task_name, task_spec, output_path)
            print(f"索引文件已生成: {path}")
            return 0
        except Exception as e:
            print(f"错误: {e}")
            return 1

    # 运行评估
    if not args.benchmark:
        parser.print_help()
        return 1

    if not args.model:
        print("错误: 必须指定 --model 参数")
        return 1

    benchmark_name, task_spec, config, task_config = resolve_task(args.benchmark, catalog)

    # 确保 .venvs 目录存在
    VENVS_DIR.mkdir(parents=True, exist_ok=True)

    return run_eval(
        benchmark_name=benchmark_name,
        task_spec=task_spec,
        config=config,
        model=args.model,
        limit=args.limit,
        judge_model=args.judge_model,
        extra_args=args.extra_args,
        dry_run=args.dry_run,
        task_config=task_config,
        no_index=args.no_index,
        index_file=args.index_file,
        api_base=args.api_base,
        api_key=args.api_key,
        log_suffix=args.log_suffix,
    )


if __name__ == "__main__":
    sys.exit(main())
