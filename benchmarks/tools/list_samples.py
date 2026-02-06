#!/usr/bin/env python3
"""
枚举数据集样本 ID

该脚本在 benchmark 的虚拟环境中运行，加载 task 并输出所有样本 ID。
无显式 ID 的样本使用 inspect_ai 自动分配的数字 ID (从 1 开始)。

用法:
    python list_samples.py <task_spec>

输出格式 (JSON):
    {"ids": ["1", "2", "3", ...], "total": 123}
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_task_module(task_spec: str):
    """
    加载 task 模块

    task_spec 格式: path/to/module.py@task_name
    """
    if "@" in task_spec:
        module_path, task_name = task_spec.rsplit("@", 1)
    else:
        module_path = task_spec
        task_name = None

    module_path = Path(module_path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("task_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["task_module"] = module
    spec.loader.exec_module(module)

    return module, task_name


def get_dataset_from_task(module, task_name: str | None):
    """
    从 task 模块获取 dataset

    尝试多种方式:
    1. 调用 task 函数，从返回的 Task 对象获取 dataset
    2. 直接调用 read_dataset 函数
    """
    from inspect_ai import Task
    from inspect_ai.dataset import Dataset

    # 尝试方式1: 从 task 函数获取
    if task_name:
        task_func = getattr(module, task_name, None)
        if task_func:
            try:
                task_obj = task_func()
                if isinstance(task_obj, Task) and task_obj.dataset:
                    return task_obj.dataset
            except Exception:
                pass

    # 尝试方式2: 直接调用 read_dataset
    read_dataset = getattr(module, "read_dataset", None)
    if read_dataset:
        try:
            return read_dataset()
        except Exception:
            pass

    # 尝试方式3: 从同目录的 dataset.py 导入
    module_dir = Path(module.__file__).parent
    dataset_path = module_dir / "dataset.py"
    if dataset_path.exists():
        ds_spec = importlib.util.spec_from_file_location("dataset_module", dataset_path)
        if ds_spec and ds_spec.loader:
            ds_module = importlib.util.module_from_spec(ds_spec)
            ds_spec.loader.exec_module(ds_module)
            read_dataset = getattr(ds_module, "read_dataset", None)
            if read_dataset:
                return read_dataset()

    raise RuntimeError("Cannot find dataset in task module")


def extract_sample_ids(dataset) -> list[str]:
    """
    提取样本 ID 列表

    inspect_ai 对于没有显式 id 的样本，会自动分配从 1 开始的数字 ID
    """
    ids = []
    for i, sample in enumerate(dataset, start=1):
        sample_id = getattr(sample, "id", None)
        if sample_id is None:
            sample_id = str(i)
        else:
            sample_id = str(sample_id)
        ids.append(sample_id)
    return ids


def main():
    if len(sys.argv) < 2:
        print("Usage: python list_samples.py <task_spec>", file=sys.stderr)
        sys.exit(1)

    task_spec = sys.argv[1]

    try:
        module, task_name = load_task_module(task_spec)
        dataset = get_dataset_from_task(module, task_name)
        ids = extract_sample_ids(dataset)

        result = {
            "ids": ids,
            "total": len(ids),
        }
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
